from dataclasses import dataclass
from typing import Iterable

from lift_bench.hierarchical_drafting.utils.file_utils import (
    makedirs,
    dump_yaml,
)

from lift_bench.hierarchical_drafting.core.base_blocks import (
    AtomicBlock,
)


@dataclass(kw_only=True)
class GenerateSeedDraftingConfigBlock(AtomicBlock):
    seed_method: str
    topics_path: str

    @property
    def ext(self) -> str:
        return "yaml"

    @property
    def output_filename(self) -> str:
        return "seed_config"

    def write_output(self, data: Iterable[dict]) -> None:
        data = list(data)
        assert len(data) == 1
        data = data[0]
        makedirs(self.task_output_dir)
        dump_yaml(data, self.output_path)
        self.logger.info(f"Output written to {self.output_path}")

    async def _run_impl(self) -> None:
        if self.seed_method != "wiki_topics":
            raise ValueError(f"Unsupported seed method: {self.seed_method}")

        configs = []

        # TODO: add constraint sampler configurations
        for i in range(1):
            config_item = {
                "config_name": f"config_{i + 1}",
                "seed_method": self.seed_method,
                "method_params": {
                    "topics_path": self.topics_path,
                },
            }
            configs.append(config_item)

        self.write_output([{"configs": configs}])
