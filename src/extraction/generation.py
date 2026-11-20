"""Streaming generation and failure-only passes without generation journals."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Callable, Iterator

from extraction.backends.vllm_workers import VllmWorkerPool


def generate_once(generate, tasks, complete):
    admitted, completed = set(), set()

    def requests():
        for task in tasks:
            if task.task_id in admitted:
                raise ValueError(f"duplicate generation task: {task.task_id}")
            admitted.add(task.task_id)
            yield task

    def receive(task_id, text, event=None):
        if task_id not in admitted:
            raise RuntimeError(f"unexpected generation result: {task_id}")
        if task_id not in completed:
            complete(task_id, text, event)
            completed.add(task_id)

    returned = generate(requests(), receive)
    for task_id in admitted - completed:
        if task_id not in returned:
            raise RuntimeError(f"missing generation result: {task_id}")
        receive(task_id, returned[task_id])


def generate_penalty_passes(generate, tasks, penalties, complete, log=None, on_pass=None):
    """Finish each pass before retrying failures; keep only pending tasks in memory.

    complete(task_id, text, event, final=...) publishes the outcome and returns whether to
    retry (anything but success, REPEAT_OCCURRED included). event is the worker result.
    """
    from dataclasses import replace
    from extraction.generation_config import penalty_schedule

    schedule = penalty_schedule(penalties)
    for index, penalty in enumerate(schedule):
        active, retry = {}, []
        final = index == len(schedule) - 1
        if on_pass:
            on_pass(index + 1, len(schedule), None if index == 0 else len(tasks))
        if log:
            log(f"[vLLM] repetition_penalty={penalty:.2f} pass={index + 1}/{len(schedule)}")

        def requests():
            for task in tasks:
                request = replace(task, repetition_penalty=penalty)
                active[request.task_id] = request
                yield request

        def receive(task_id, text, event=None):
            task = active.pop(task_id)
            if complete(task_id, text, event, final=final) and not final:
                retry.append(task)

        generate_once(generate, requests(), receive)
        if not retry:
            break
        tasks = retry


@contextmanager
def vllm_generator(
    *, model_path: Path, settings=None, image_limit=6, log=None, on_progress=None
) -> Iterator[Callable]:
    """Yield generate(tasks, on_task_complete); on_task_complete(task_id, text, event)."""

    def ready(event):
        if log:
            log(
                f"[vLLM] GPU {event['gpu_id']} ready | {event['model_family']} | {event['gpu_name']} | settings={event['settings']}"
            )

    with ExitStack() as resources:
        pool = None

        def generate(tasks, on_task_complete=None):
            nonlocal pool
            if pool is None:
                if log:
                    log("[vLLM] starting GPU workers")
                pool = resources.enter_context(
                    VllmWorkerPool(
                        str(model_path),
                        settings=settings,
                        image_limit=image_limit,
                        on_runtime=ready,
                        on_progress=on_progress,
                    )
                )

            def complete(task_id, text):
                on_task_complete(task_id, text, pool.last_result)

            return pool.generate(tasks, complete if on_task_complete else None)

        yield generate
