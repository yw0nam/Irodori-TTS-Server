from __future__ import annotations

import asyncio

from irodori_openai_tts.batching import SynthesisBatcher
from irodori_tts.inference_runtime import SamplingRequest

RUNTIME = object()


class FakeRunBatch:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def __call__(self, runtime, requests):
        assert runtime is RUNTIME
        self.batches.append([request.text for request in requests])
        return [f"result:{request.text}" for request in requests]


def make_batcher(run_batch, *, max_batch_size=8, wait_seconds=0.01, queue_timeout=5.0):
    return SynthesisBatcher(
        max_batch_size=max_batch_size,
        wait_seconds=wait_seconds,
        queue_timeout=queue_timeout,
        run_batch=run_batch,
    )


def submit_all_concurrently(batcher, requests):
    async def main():
        return await asyncio.gather(*(batcher.submit(RUNTIME, request) for request in requests))

    return asyncio.run(main())


def test_compatible_submits_share_one_batch_in_submit_order():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    results = submit_all_concurrently(
        batcher,
        [SamplingRequest(text="a"), SamplingRequest(text="b"), SamplingRequest(text="c")],
    )

    assert run_batch.batches == [["a", "b", "c"]]
    assert results == ["result:a", "result:b", "result:c"]


def test_incompatible_submits_run_in_separate_batches():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    results = submit_all_concurrently(
        batcher,
        [
            SamplingRequest(text="a", num_steps=10),
            SamplingRequest(text="b", num_steps=20),
            SamplingRequest(text="c", num_steps=10),
        ],
    )

    assert run_batch.batches == [["a", "c"], ["b"]]
    assert results == ["result:a", "result:b", "result:c"]


def test_batches_are_capped_at_max_batch_size():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch, max_batch_size=2)

    submit_all_concurrently(
        batcher,
        [SamplingRequest(text="a"), SamplingRequest(text="b"), SamplingRequest(text="c")],
    )

    assert run_batch.batches == [["a", "b"], ["c"]]


def test_request_without_compatibility_key_runs_alone():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    submit_all_concurrently(
        batcher,
        [
            SamplingRequest(text="a", speaker_uncond_mode="noise"),
            SamplingRequest(text="b", speaker_uncond_mode="noise"),
        ],
    )

    assert run_batch.batches == [["a"], ["b"]]
