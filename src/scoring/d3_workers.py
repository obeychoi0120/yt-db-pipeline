"""One d3 decision model per GPU; each request is one scene (or one graph text)."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import traceback


def load_model(model_path: str, *, batch_size: int = 8):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("scoring requires transformers; pip install -r requirements-scoring.txt") from exc
    return AutoModel.from_pretrained(model_path, trust_remote_code=True, batch_size=batch_size)


def _worker_main(worker_index, gpu_id, model_path, batch_size, task_queue, result_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        model = load_model(model_path, batch_size=batch_size)
        result_queue.put({"kind": "ready", "worker_index": worker_index, "gpu_id": gpu_id})
        while True:
            task = task_queue.get()
            if task is None:
                return
            try:
                result = model.system_one(
                    state=task["state"], questions=task["questions"], images=task.get("images")
                )
                result_queue.put({
                    "kind": "result", "task_id": task["task_id"], "answers": result["answers"],
                    "input_tokens": result.get("usage", {}).get("input_tokens"),
                })
            except Exception as exc:  # one bad scene must not stop the worker
                result_queue.put({
                    "kind": "result", "task_id": task["task_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
    except BaseException:
        result_queue.put({"kind": "failed", "worker_index": worker_index, "gpu_id": gpu_id,
                          "error": traceback.format_exc()})


def visible_gpu_ids() -> list[str]:
    import torch

    count = int(torch.cuda.device_count())
    if count <= 0:
        raise RuntimeError("scoring requires at least one visible CUDA GPU")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [str(i) for i in range(count)] if mask is None else [
        value.strip() for value in mask.split(",")
    ][:count]


class D3WorkerPool:
    def __init__(self, model_path, *, batch_size=8, log=print):
        self._context = mp.get_context("spawn")
        self._tasks = self._context.Queue()
        self._results = self._context.Queue()
        self._log = log
        self.gpu_ids = visible_gpu_ids()
        self._processes = [
            self._context.Process(
                target=_worker_main,
                args=(index, gpu_id, str(model_path), batch_size, self._tasks, self._results),
            )
            for index, gpu_id in enumerate(self.gpu_ids)
        ]
        for process in self._processes:
            process.start()

    def run(self, tasks, on_result):
        """Feed tasks with at most two in flight per GPU; on_result(event) per finished task."""
        iterator = iter(tasks)
        inflight, exhausted = 0, False
        capacity = 2 * len(self._processes)
        while True:
            while not exhausted and inflight < capacity:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    self._tasks.put(task)
                    inflight += 1
            if exhausted and inflight == 0:
                return
            try:
                event = self._results.get(timeout=1.0)
            except queue.Empty:
                if any(not process.is_alive() for process in self._processes):
                    raise RuntimeError("a d3 worker exited before finishing its tasks")
                continue
            if event["kind"] == "ready":
                self._log(f"[d3] GPU {event['gpu_id']} ready")
            elif event["kind"] == "failed":
                raise RuntimeError(f"d3 worker on GPU {event['gpu_id']} failed: {event['error']}")
            else:
                inflight -= 1
                on_result(event)

    def close(self):
        for _ in self._processes:
            self._tasks.put(None)
        for process in self._processes:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            for process in self._processes:
                process.terminate()
