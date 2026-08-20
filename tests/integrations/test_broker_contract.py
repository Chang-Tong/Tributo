"""Contract tests for the transport-neutral Broker API v1."""

from __future__ import annotations

from tributo.integrations.broker import (
    BROKER_API_VERSION,
    BrokerPlugin,
    BrokerRuntime,
    CancellationSpec,
    JobResult,
    Message,
    TaskConsumer,
    TaskDisposition,
    TaskOutcome,
)


class _LegacyConsumer(TaskConsumer):
    """A pre-v1 consumer implementing only the original ABC methods."""

    def poll(self, timeout_ms: int = 5000) -> Message | None:
        return None

    def ack(self, message: Message) -> None:
        pass


def test_old_consumer_subclasses_remain_instantiable() -> None:
    consumer = _LegacyConsumer()
    message = Message("job-1", {})
    consumer.retry(message, "temporary")
    assert consumer.recover_pending() == 0
    consumer.close()


def test_provider_v1_data_contract_matches_external_provider() -> None:
    message = Message(
        job_id=None,
        payload={"raw": "{}"},
        delivery_id="7-0",
        delivery_attempt=2,
    )
    result = JobResult(
        job_id="job-1",
        status="accepted",
        run_id="job-1",
        attempt_id="attempt-1",
        execution_id="ray-job",
        submission_id="submission-1",
        artifact_refs=[{"name": "onnx-model", "format": "onnx"}],
        bundle_id="bundle-1",
        bundle_uri="/models/bundle-1",
        manifest_uri="/models/bundle-1/manifest.json",
    )
    spec = CancellationSpec(
        broker_id="knova-redis",
        job_id="job-1",
        options={"config_env": "TRIBUTO_BROKER_CONFIG_JSON"},
    )

    assert BROKER_API_VERSION == 1
    assert message.delivery_id == "7-0"
    assert message.delivery_attempt == 2
    assert result.execution_id == "ray-job"
    assert result.bundle_id == "bundle-1"
    assert spec.as_dict()["broker_id"] == "knova-redis"
    assert TaskOutcome(TaskDisposition.ACK).disposition is TaskDisposition.ACK
    assert issubclass(BrokerPlugin, object)
    assert issubclass(BrokerRuntime, object)
