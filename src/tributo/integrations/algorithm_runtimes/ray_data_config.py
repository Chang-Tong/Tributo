"""Ray Train DataConfig compatibility for exact row coverage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from ray.data import ExecutionResources
from ray.train import DataConfig

from tributo.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.data import DataIterator, Dataset, ExecutionOptions, NodeIdStr


class _Ray255Config(Protocol):
    _num_train_cpus: float
    _num_train_gpus: float

    def _get_execution_options(self, dataset_name: str) -> ExecutionOptions: ...
    def _is_v2_autoscaler(self) -> bool: ...


@DeveloperAPI
class ExactCoverageDataConfig(DataConfig):
    """Keep every input row when Ray Train assigns Dataset iterators."""

    def _shard_options(self, name: str) -> ExecutionOptions:
        resolve = getattr(self, "_resolve_execution_options", None)
        if resolve is not None:
            return cast("ExecutionOptions", resolve(name))
        # Ray 2.55.1 assigns V1 autoscaler reservations in DataConfig.configure.
        config = cast(_Ray255Config, self)
        options = config._get_execution_options(name)
        if options.is_resource_limits_default() and not config._is_v2_autoscaler():
            options.exclude_resources = options.exclude_resources.add(
                ExecutionResources(
                    cpu=config._num_train_cpus, gpu=config._num_train_gpus
                )
            )
        return options

    @DeveloperAPI
    def configure(
        self,
        datasets: dict[str, Dataset],
        world_size: int,
        worker_handles: list[ActorHandle] | None,
        worker_node_ids: list[NodeIdStr] | None,
        **kwargs: Any,
    ) -> list[dict[str, DataIterator]]:
        """Assign Train shards with ``equal=False``."""
        del worker_handles, kwargs
        output: list[dict[str, DataIterator]] = [{} for _ in range(world_size)]
        for dataset_name, dataset in datasets.items():
            if dataset.name is None:
                dataset.set_name(dataset_name)

        datasets_to_split = (
            set(datasets)
            if self._datasets_to_split == "all"
            else set(self._datasets_to_split)
        )
        locality_hints = worker_node_ids if self._enable_shard_locality else None
        for name, dataset in datasets.items():
            configured = dataset.copy(dataset)
            configured.context.execution_options = self._shard_options(name)
            if name in datasets_to_split:
                splits = configured.streaming_split(
                    world_size,
                    equal=False,
                    locality_hints=locality_hints,
                )
                for rank, split in enumerate(splits):
                    output[rank][name] = split
            else:
                for rank in range(world_size):
                    output[rank][name] = configured.iterator()
        return output


@DeveloperAPI
class TorchRoleDataConfig(ExactCoverageDataConfig):
    """Route Torch roles independently: split_exact roles shard, replicate roles do not."""

    def __init__(self, routes: Mapping[str, object], **kwargs: Any) -> None:
        split_roles = [
            role
            for role, route in routes.items()
            if getattr(route, "mode", None) == "split_exact"
        ]
        super().__init__(datasets_to_split=split_roles, **kwargs)


__all__ = ["ExactCoverageDataConfig", "TorchRoleDataConfig"]
