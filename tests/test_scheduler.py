"""scheduler 队列守恒和 batching 行为的回归测试。"""

import sys
import os

# 让 pytest 在无需安装包的情况下导入本地 src-layout 包。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from collections import deque
from unittest.mock import MagicMock
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=4,
):
    """创建一个限制项容易覆盖的小型 scheduler。"""
    return Scheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
    )


def inject_running(scheduler: Scheduler, *seqs: Sequence):
    """绕过 prefill 分配，直接把序列放进 running 队列。"""
    for seq in seqs:
        # 测试会用 mock 替换 block_manager，因此不需要真实 block table。
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)


def all_tracked(scheduler: Scheduler, scheduled: list[Sequence]) -> set:
    """返回 scheduler 当前能追踪到的全部序列集合。"""
    # 一个序列只要还在队列中、正在运行，或在当前 scheduled batch 中，就视为仍被追踪。
    return set(scheduler.running) | set(scheduler.waiting) | set(scheduled)


class TestBug2TokenLimitBreak:
    """
    场景：running 中有 3 个序列，且 can_append 全部为 True。
          max_num_batched_tokens=2，因此每轮只能放下 2 个。
    预期：schedule() 后 seq_c 仍应留在 running 中。
    错误行为：seq_c 被 popleft 后触发 limit 和 break，却没有被恢复，导致永久丢失。
    """

    def _run(self, scheduler: Scheduler):
        """构造该 bug 类共享的三序列场景。"""
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        seq_c = Sequence([7, 8, 9])
        inject_running(scheduler, seq_a, seq_b, seq_c)

        scheduler.block_manager = MagicMock()
        # 强制所有序列都可 append，使测试只关注 batch limit。
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        return seq_a, seq_b, seq_c, scheduled, is_prefill

    def test_seq_count_is_correct(self):
        """token budget 应限制 batch，但不能丢掉下一个序列。"""
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        # 当前 batch 只能容纳两个单 token decode step。
        assert len(scheduled) == 2

        # 调度结束后，未被调度的第三个序列必须仍在队列中。
        assert seq_c in scheduler.running, (
            "Bug 2: seq_c was popleft-ed and the break fired before it could be "
            "added to scheduled_sequences or put back into self.running → LOST"
        )

    def test_seq_count_limit_variant(self):
        """同一个 bug，只是由 max_num_sequences 而不是 token budget 触发。"""
        scheduler = make_scheduler(max_num_sequences=2, max_num_batched_tokens=100)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        assert len(scheduled) == 2

        assert seq_c in scheduler.running, (
            "Bug 2 (seq-count variant): seq_c lost when len(scheduled_sequences) "
            ">= max_num_sequences caused the break"
        )

    def test_no_sequence_is_lost(self):
        """序列全集必须保持守恒。"""
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        for seq in (seq_a, seq_b, seq_c):
            assert seq in tracked, f"seq {seq.seq_id} disappeared from the scheduler"


class TestBug1CanAppendFailure:
    """
    场景：running 中有 2 个序列，第一个序列 can_append 返回 False。
    预期：seq_a 要么放回 running 以后重试，要么被抢占进 waiting；
          总之不能完全消失。
    错误行为：seq_a 被 popleft 后 can_append 失败，代码执行
             self.preempt(self.running.pop()) 抢占了 seq_b，
             但 seq_a 本身没有被处理，导致丢失。
    """

    def _run(self, scheduler: Scheduler):
        """构造该 bug 类共享的双序列抢占场景。"""
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        # 第一次调用对应 seq_a，不能 append；后续调用都可以 append。
        mock_bm.can_append.side_effect = [False, True, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        return seq_a, seq_b, scheduled, is_prefill

    def test_seq_a_not_lost(self):
        """can_append 失败的序列仍必须被某个队列追踪。"""
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, (
            "Bug 1: seq_a was popleft-ed, can_append returned False, "
            "self.preempt(self.running.pop()) preempted seq_b instead, "
            "and seq_a was never restored → LOST"
        )

    def test_total_conservation(self):
        """两个序列都不能消失。"""
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, f"seq_a disappeared"
        assert seq_b in tracked, f"seq_b disappeared"


class TestSchedulerHappyPath:
    def test_prefill_scheduled_first(self):
        """waiting prompt 应先进入 prefill，再选择 decode 工作。"""
        scheduler = make_scheduler(max_num_batched_tokens=100, max_cached_blocks=50)
        seq = Sequence([1, 2, 3, 4])
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert seq in scheduled
        assert seq in scheduler.running

    def test_all_running_seqs_scheduled_when_budget_allows(self):
        """budget 足够时，所有 running 序列都应进入 decode。"""
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_a = Sequence([1])
        seq_b = Sequence([2])
        inject_running(scheduler, seq_a, seq_b)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 2
        # decode batch 会重新插回 running，以便进行下一步 decode。
        assert seq_a in scheduler.running
        assert seq_b in scheduler.running

    def test_preempt_only_seq_when_cant_append_and_running_empty(self):
        """唯一 running 序列如果不能 append，应被抢占回 waiting。"""
        scheduler = make_scheduler()
        seq = Sequence([1, 2])
        inject_running(scheduler, seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = False
        scheduler.block_manager.deallocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 0
        assert seq in scheduler.waiting
        assert seq.status == SequenceStatus.WAITING
