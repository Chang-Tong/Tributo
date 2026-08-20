"""Tests for the generic Broker runner lifecycle."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

from tributo.integrations.broker import Message, TaskDisposition, TaskOutcome
from tributo.integrations.broker_runner import BrokerRunner, BrokerRunnerState


def _runner(
    *outcomes: TaskOutcome,
) -> tuple[BrokerRunner, MagicMock, MagicMock, MagicMock, Message]:
    message = Message("job-1", {}, delivery_id="7-0")
    consumer = MagicMock()
    consumer.poll.side_effect = [message] * len(outcomes)
    consumer.recover_pending.return_value = 0
    runtime = MagicMock()
    runtime.consumer = consumer
    runtime.handle.side_effect = deque(outcomes)
    plugin = MagicMock()
    plugin.broker_id = "fake"
    plugin.create_runtime.return_value = runtime
    runner = BrokerRunner(plugin, {}, sleep=lambda _delay: None)
    return runner, plugin, runtime, consumer, message


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_runner_recovers_pending_then_polls_handles_and_acks() -> None:
    runner, _plugin, runtime, consumer, message = _runner(
        TaskOutcome(TaskDisposition.ACK)
    )

    assert runner.run_once() is True

    consumer.recover_pending.assert_called_once_with()
    consumer.poll.assert_called_once_with(5000)
    runtime.handle.assert_called_once_with(message)
    consumer.ack.assert_called_once_with(message)
    consumer.retry.assert_not_called()
    assert runner.state is BrokerRunnerState.RUNNING


def test_runner_ticks_runtime_maintenance_before_message_poll() -> None:
    runner, _plugin, runtime, consumer, _message = _runner(
        TaskOutcome(TaskDisposition.ACK)
    )

    assert runner.run_once() is True

    runtime.maintain.assert_called_once_with()
    consumer.poll.assert_called_once_with(5000)


def test_runner_ticks_runtime_maintenance_during_idle_poll() -> None:
    runner, _plugin, runtime, consumer, _message = _runner()
    consumer.poll.side_effect = None
    consumer.poll.return_value = None

    assert runner.run_once() is False

    runtime.maintain.assert_called_once_with()
    consumer.poll.assert_called_once_with(5000)


def test_maintenance_failure_reconnects_without_polling() -> None:
    runner, _plugin, runtime, consumer, _message = _runner()
    consumer.poll.side_effect = None
    runtime.maintain.side_effect = ConnectionError("supervisor unavailable")

    assert runner.run_once() is False

    consumer.poll.assert_not_called()
    runtime.close.assert_called_once_with()
    assert runner.state is BrokerRunnerState.RECONNECTING


def test_runner_leaves_retry_pending() -> None:
    runner, _plugin, _runtime, consumer, message = _runner(
        TaskOutcome(TaskDisposition.RETRY, error="Ray unavailable")
    )

    assert runner.run_once() is True

    consumer.retry.assert_called_once_with(message, "Ray unavailable")
    consumer.ack.assert_not_called()


def test_runner_periodically_recovers_and_reexecutes_retry_in_same_process() -> None:
    clock = _Clock()
    message = Message("job-1", {}, delivery_id="7-0")
    consumer = MagicMock()
    consumer.recover_pending.return_value = 0
    consumer.poll.side_effect = [message, None, message]
    runtime = MagicMock()
    runtime.consumer = consumer
    runtime.handle.side_effect = [
        TaskOutcome(TaskDisposition.RETRY, error="Ray unavailable"),
        TaskOutcome(TaskDisposition.ACK),
    ]
    plugin = MagicMock()
    plugin.broker_id = "fake"
    plugin.create_runtime.return_value = runtime
    runner = BrokerRunner(
        plugin,
        {},
        pending_recovery_interval=10,
        clock=clock,
        sleep=lambda _delay: None,
    )

    assert runner.run_once() is True
    clock.advance(9)
    assert runner.run_once() is False
    assert consumer.recover_pending.call_count == 1

    clock.advance(1)
    assert runner.run_once() is True
    assert consumer.recover_pending.call_count == 2
    assert runtime.handle.call_count == 2
    consumer.retry.assert_called_once_with(message, "Ray unavailable")
    consumer.ack.assert_called_once_with(message)


def test_pending_recovery_drains_multiple_batches_with_a_bound() -> None:
    runner, _plugin, _runtime, consumer, _message = _runner()
    runner._max_recovery_batches = 3
    consumer.recover_pending.return_value = 2

    runner.start()

    assert consumer.recover_pending.call_count == 3


def test_transport_failure_closes_runtime_and_backs_off_before_reconnect() -> None:
    sleeps: list[float] = []
    runner, plugin, runtime, consumer, _message = _runner()
    runner._sleep = sleeps.append
    runner._backoff_initial = 0.25
    runner._backoff_max = 1.0
    runner._next_backoff = 0.25
    consumer.poll.side_effect = ConnectionError("broker unavailable")

    assert runner.run_once() is False

    assert runner.state is BrokerRunnerState.RECONNECTING
    runtime.close.assert_called_once_with()
    assert sleeps == [0.25]

    replacement = MagicMock()
    replacement.consumer.poll.return_value = None
    replacement.consumer.recover_pending.return_value = 0
    plugin.create_runtime.return_value = replacement
    assert runner.run_once() is False
    assert runner.state is BrokerRunnerState.RUNNING


def test_consecutive_ack_failures_use_exponential_backoff() -> None:
    sleeps: list[float] = []
    message = Message("job-1", {}, delivery_id="7-0")
    runtimes = []
    for _ in range(3):
        consumer = MagicMock()
        consumer.recover_pending.return_value = 0
        consumer.poll.return_value = message
        consumer.ack.side_effect = ConnectionError("broker unavailable")
        runtime = MagicMock()
        runtime.consumer = consumer
        runtime.handle.return_value = TaskOutcome(TaskDisposition.ACK)
        runtimes.append(runtime)
    plugin = MagicMock()
    plugin.broker_id = "fake"
    plugin.create_runtime.side_effect = runtimes
    runner = BrokerRunner(
        plugin,
        {},
        backoff_initial=0.25,
        backoff_max=1.0,
        sleep=sleeps.append,
    )

    assert runner.run_once() is True
    assert runner.run_once() is True
    assert runner.run_once() is True

    assert sleeps == [0.25, 0.5, 1.0]
    assert runner.state is BrokerRunnerState.RECONNECTING


def test_close_is_idempotent() -> None:
    runner, _plugin, runtime, _consumer, _message = _runner()
    runner.start()
    runner.close()
    runner.close()
    runtime.close.assert_called_once_with()
    assert runner.state is BrokerRunnerState.CLOSED


def test_close_failure_still_closes_runner() -> None:
    runner, _plugin, runtime, _consumer, _message = _runner()
    runtime.close.side_effect = ConnectionError("already disconnected")
    runner.start()

    runner.close()

    assert runner.state is BrokerRunnerState.CLOSED
