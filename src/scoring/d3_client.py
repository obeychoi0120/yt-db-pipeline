"""The d3 judge behind an HTTP System One server (vLLM Semantic Router Engine mode).

    vllm-sr serve vllm-sr/d3-mini -e -dp 4 --device-ids 0,1,2,3

serves ``POST /v1/systemone`` in the vllm-sr/5.0 schema: keyframes go as ``assets`` whose
placeholders stand in front of the state, which d3 reads as the images of that state. The
reply has the same ``answers`` as ``system_one`` in process. Many scenes are sent at once so
the server can batch them; the reply goes to the same ``on_result`` as D3WorkerPool's.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

SYSTEM_ONE_PATH = "/v1/systemone"
# Busy replies are retried: vllm-sr answers 529 when overloaded and 503 while starting.
# A 504 already waited the whole deadline and is not retried; rerunning scores the scene.
RETRY_STATUSES = {429, 502, 503, 529}
# vllm-sr gives up on a request after 30 s unless options.deadline_ms says otherwise; queued
# behind many scenes, a request needs longer. 600000 ms is the listener's default maximum.
DEADLINE_MS = 600_000
MAX_RETRIES = 8


def image_asset(path: str | Path) -> dict:
    """One vllm-sr/5.0 image asset; the server detects png/jpeg/webp from the bytes."""
    return {"type": "image", "data_base64": base64.b64encode(Path(path).read_bytes()).decode("ascii")}


def placed_request(model: str, state, questions: dict, images=None) -> dict:
    """A vllm-sr/5.0 request: the images' placeholders in front of the state."""
    payload = {"model": model, "state": state, "questions": questions}
    if images:
        keys = [f"\u27e6img-{index}\u27e7" for index in range(1, len(images) + 1)]
        payload["state"] = [*keys, state]
        payload["assets"] = {key: image_asset(path) for key, path in zip(keys, images)}
    return payload


class D3ServerPool:
    def __init__(self, url: str, *, model: str, concurrency: int = 32,
                 deadline_ms: int = DEADLINE_MS, log=print):
        self.url = url.rstrip("/") + SYSTEM_ONE_PATH
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.deadline_ms = int(deadline_ms)
        # Wait a little past the server's deadline so its 504 arrives instead of a socket timeout.
        self.timeout = self.deadline_ms / 1000 + 30
        self._log = log
        # The server is local or on the intranet: never send it through HTTP(S)_PROXY.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._executor = ThreadPoolExecutor(max_workers=self.concurrency)

    def _post(self, body: bytes) -> dict:
        request = urllib.request.Request(
            self.url, data=body, headers={"content-type": "application/json"}, method="POST"
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def _ask(self, task: dict) -> dict:
        payload = placed_request(self.model, task["state"], task["questions"], task.get("images"))
        payload["options"] = {"deadline_ms": self.deadline_ms}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        delay = 1.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                reply = self._post(body)
                return {
                    "kind": "result", "task_id": task["task_id"], "answers": reply["answers"],
                    "input_tokens": (reply.get("usage") or {}).get("input_tokens"),
                }
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                    return {"kind": "result", "task_id": task["task_id"],
                            "error": f"HTTP {exc.code}: {detail}"}
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == MAX_RETRIES:
                    return {"kind": "result", "task_id": task["task_id"],
                            "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(delay)
            delay = min(delay * 2, 30)
        raise AssertionError("unreachable")

    def check(self) -> None:
        """Fail at once with a clear message when nothing listens at the URL."""
        probe = {"model": self.model, "state": "ping",
                 "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}}
        try:
            self._post(json.dumps(probe).encode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"d3 server at {self.url} answered HTTP {exc.code}: "
                               f"{exc.read().decode('utf-8', 'replace')[:300]}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(f"no d3 server at {self.url} ({exc}); start it with "
                               "vllm-sr serve vllm-sr/d3-mini -e") from exc
        self._log(f"[d3] server ready: {self.url} ({self.concurrency} requests in flight)")

    def run(self, tasks, on_result):
        """Keep `concurrency` requests in flight; on_result(event) per finished task."""
        self.check()
        iterator = iter(tasks)
        inflight = set()
        while True:
            while len(inflight) < self.concurrency:
                task = next(iterator, None)
                if task is None:
                    break
                inflight.add(self._executor.submit(self._ask, task))
            if not inflight:
                return
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                on_result(future.result())

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._executor.shutdown(wait=exc_type is None, cancel_futures=True)


def open_judge(context, settings: dict, *, server: str | None, log=print):
    """The d3 judge: a vllm-sr server when --d3-server is given, else one model per GPU."""
    if server:
        return D3ServerPool(server, model=settings["server_model"],
                            concurrency=settings["server_concurrency"], log=log)
    from scoring.d3_workers import D3WorkerPool

    return D3WorkerPool(context.step_path("scoring", "d3_path"), batch_size=settings["batch_size"], log=log)


def judge_name(context, settings: dict, *, server: str | None) -> str:
    """Part of the scoring identity: the same d3 weights reuse their scores on either backend."""
    if server and not settings.get("d3_path"):
        return settings["server_model"]
    return str(context.step_path("scoring", "d3_path"))
