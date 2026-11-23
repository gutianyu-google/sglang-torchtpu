"""SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE defers only mid-prefill chunk inserts.

The insert at the end of a prefill must still run: a follow-up turn diverges right
after the prompt, and its match needs the SWA window that insert keeps.
"""

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from swa_cache_world import PAGE, SWA_POOL, WINDOW, KVWorld

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Tree:
    def __init__(self, defers=True):
        self.defers = defers
        self.calls = []

    def defer_chunk_checkpoint(self, req):
        self.calls.append(("defer", req.extend_end))
        return self.defers

    def checkpoint(self, req, *, up_to):
        self.calls.append(("checkpoint", up_to))


def _run(stride, mid_prefill, start=4096, end=8192, tree=None):
    req = SimpleNamespace(
        rid="r",
        finished=lambda: False,
        skip_radix_cache_insert=False,
        prefix_len=start,
        extend_end=end,
    )
    tree = tree or _Tree()
    with envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride):
        checkpoint_kv_cache(req, tree, mid_prefill=mid_prefill)
    return tree.calls


class TestDeferChunkInsert(unittest.TestCase):
    def test_mid_prefill_chunk_is_deferred(self):
        self.assertEqual(_run(16384, mid_prefill=True), [("defer", 8192)])

    def test_end_of_prefill_insert_runs(self):
        self.assertEqual(_run(16384, mid_prefill=False), [("checkpoint", 8192)])

    def test_zero_stride_inserts_every_chunk(self):
        self.assertEqual(_run(0, mid_prefill=True), [("checkpoint", 8192)])

    def test_stride_inserts_the_chunk_that_reaches_a_multiple(self):
        # 16K stride, 4K chunks: [12K, 16K) and [30K, 34K) reach a multiple,
        # [16K, 20K) does not.
        self.assertEqual(_run(16384, True, 12288, 16384), [("checkpoint", 16384)])
        self.assertEqual(_run(16384, True, 30720, 34816), [("checkpoint", 34816)])
        self.assertEqual(_run(16384, True, 16384, 20480), [("defer", 20480)])

    def test_cache_that_declines_still_inserts(self):
        calls = _run(16384, mid_prefill=True, tree=_Tree(defers=False))
        self.assertEqual(calls, [("defer", 8192), ("checkpoint", 8192)])


def _swa_cache(**overrides):
    attrs = dict(
        disable=False,
        supports_swa=lambda: True,
        supports_mamba=lambda: False,
        page_size=4,
        _components_tuple=(Mock(),),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int32).reshape(2, 32)
        ),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _req(**overrides):
    attrs = dict(
        session_id=None,
        finished=lambda: False,
        full_untruncated_fill_ids=list(range(30)),
        kv=SimpleNamespace(cache_protected_len=0, cache_inserted_len=0, req_pool_idx=1),
        priority=None,
        kv_rotation_base=None,
        prefix_len=0,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestUnifiedDeferChunkCheckpoint(unittest.TestCase):
    def test_falls_back_outside_swa_only_trees(self):
        for cache, req in (
            (_swa_cache(supports_swa=lambda: False), _req()),
            (_swa_cache(supports_mamba=lambda: True), _req()),
            (_swa_cache(disable=True), _req()),
            (_swa_cache(), _req(session_id="s")),
        ):
            self.assertFalse(UnifiedRadixCache.defer_chunk_checkpoint(cache, req))
            cache._components_tuple[0].free_out_of_window_slots.assert_not_called()
            self.assertEqual(req.prefix_len, 0)


class _PrefillWorld(KVWorld):
    """A real hybrid-SWA UnifiedRadixCache that chunk-prefills a prompt as the
    scheduler does: match and lock at admission, checkpoint_kv_cache after every
    chunk but the last."""

    def prefill(self, prompt: list[int], *, chunk: int) -> tuple[int, int]:
        """Return the most SWA tokens the request held that no eviction could
        free, and the longest prefix a mid-prefill checkpoint published."""
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.rid += 1
        self.req_to_token_pool.alloc([req])
        req.init_next_round_input(tree_cache=self.cache)
        req.lock = self.cache.lock(req.last_node)
        row, pos = req.kv.req_pool_idx, req.prefix_len
        held = published = 0
        while pos < len(prompt):
            end = min(pos + chunk, len(prompt))
            req.prefix_len, req.extend_end = pos, end
            self.req_to_token_pool.write(
                (row, slice(pos, end)), self._alloc(prompt[pos:end], pos)
            )
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(prompt):
                checkpoint_kv_cache(req, self.cache, mid_prefill=True)
                req.init_next_round_input()
                published = max(published, self.match(prompt).device_prefix_len)
            swa_used = self.allocator.size_swa - self.allocator.swa_available_size()
            held = max(held, swa_used - self.cache.window_evictable_size(SWA_POOL))
        return held, published


class TestDeferredChunkSwaBound(unittest.TestCase):
    CHUNK = 8 * PAGE

    def test_long_prefill_keeps_swa_within_window_and_chunk(self):
        """A prompt three times the SWA pool must prefill in chunks, holding no more
        unevictable SWA than a window and a chunk, and leave room for the next
        request, whether mid-prefill inserts are deferred or not and whether or not
        out-of-window slots are freed. Otherwise the pool fills with the request's
        own rows and the scheduler can neither continue nor admit anything."""
        rng = random.Random(0)
        for stride in (0, 1 << 20):
            for free_out_of_window in (False, True):
                with (
                    self.subTest(stride=stride, free=free_out_of_window),
                    envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride),
                    envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(
                        free_out_of_window
                    ),
                ):
                    w = _PrefillWorld(full_tokens=16384, swa_tokens=1024)
                    prompt = [rng.randrange(1, 50000) for _ in range(3000)]
                    held, published = w.prefill(prompt, chunk=self.CHUNK)
                    self.assertLessEqual(held, WINDOW + 1 + self.CHUNK)
                    # The stride defers only when the deferred chunks free their tail.
                    deferred = stride > 0 and free_out_of_window
                    self.assertEqual(published == 0, deferred)
                    w.prefill(
                        [rng.randrange(1, 50000) for _ in range(600)], chunk=self.CHUNK
                    )
                    w.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
