import asyncio
import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from lift_bench.hierarchical_drafting.core.base_blocks import (
    ConfigBlock,
)


@dataclass(kw_only=True)
class SeedTopicsBlock(ConfigBlock):
    max_items: int = 0

    def _get_wiki_topics(self, params: dict) -> list[dict]:
        path = params["topics_path"]
        self.logger.info(f"Loading document topics from {path}...")
        df = pd.read_json(path, lines=True)
        topics_ser = pd.Series(df.title[0].values())
        topics_dedup_ser = topics_ser.drop_duplicates()
        self.logger.info(f"Loaded {len(topics_dedup_ser)} topics.")
        items = [{"topic": t} for t in topics_dedup_ser.tolist()]
        if self.max_items > 0:
            items = items[: self.max_items]
        return items

    async def run_single_config(self, config_item: dict) -> list[dict]:
        match config_item["seed_method"]:
            case "wiki_topics":
                items = self._get_wiki_topics(config_item["method_params"])
            case _:
                raise ValueError(f"Unknown seed method: {config_item['seed_method']}")
        # Enrich items with additional metadata
        for item in items:
            # TODO ignore some fields here? is the implied consistency important?
            item["example_id"] = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()

        return items

    async def _run_impl(self) -> None:
        config_futures = [self.run_single_config(config_item) for config_item in self._configs["configs"]]
        # TODO Limit concurrency?
        items_list: list[list[dict]] = await asyncio.gather(*config_futures)
        items: list[dict] = [item for sublist in items_list for item in sublist]
        self.write_output(items)
