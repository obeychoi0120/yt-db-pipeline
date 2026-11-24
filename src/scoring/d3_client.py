"""The d3 judge behind an HTTP System One server (vLLM Semantic Router Engine mode).

    vllm-sr serve vllm-sr/d3-mini -e -dp 4 --device-ids 0,1,2,3

serves ``POST /v1/systemone`` with the same request and reply as ``system_one`` in process
(the package's own ``d3_server.py`` serves the same API). Many scenes are sent at once so
the server can batch them; the reply goes to the same ``on_result`` as D3WorkerPool's.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

SYSTEM_ONE_PATH = "/v1/systemone"
# Busy replies are retried; vllm-sr answers 503 when more requests are in flight than it admits.
RETRY_STATUSES = {429, 502, 503, 504}
MAX_RETRIES = 8


def data_url(path: str | Path) -> str:
    kind = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{kind};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


class D3ServerPool:
    def __init__(self, url: str, *, model: str, concurrency: int = 32, timeout: float = 600,
                 log=print):
        self.url = url.rstrip("/") + SYSTEM_ONE_PATH
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.timeout = timeout
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
        payload = {"model": self.model, "state": task["state"], "questions": task["questions"]}
        if task.get("images"):
            payload["images"] = [data_url(path) for path in task["images"]]
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

    return D3WorkerPool(context.model_path("d3"), batch_size=settings["batch_size"], log=log)


def judge_name(context, settings: dict, *, server: str | None) -> str:
    """Part of the scoring identity: the same d3 weights reuse their scores on either backend."""
    if server and not (context.config.get("models") or {}).get("d3"):
        return settings["server_model"]
    return str(context.model_path("d3"))
