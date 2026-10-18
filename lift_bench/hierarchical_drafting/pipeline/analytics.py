import logging

import pandas as pd
from lift_bench.hierarchical_drafting.utils.file_utils import iter_json_lines
from collections import defaultdict
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode


def visualize_data(data_path: str, logger: logging.Logger, output_file: str | None = None):
    stats = []

    for item in iter_json_lines(data_path):
        stat = {}
        node = DocumentNode.from_md(item["document_outline"])

        # document size
        stat["leaf_count"] = sum([c.is_leaf() for c in node.get_children(recursive=True)])
        stat["doc_depth"] = node.get_height()
        stat["doc_length"] = len(node.get_children())

        # constraint count stats
        ctype_counts_dct = defaultdict(int)
        cname_counts_dct = defaultdict(int)
        for k, v in item.items():
            if not k.startswith("sampled_"):
                continue
            ctype = k[len("sampled_") :]
            for hname, constraint_dct in v.items():
                for cname, clst in constraint_dct.items():
                    if cname == "doc_structure":
                        continue
                    ctype_counts_dct[ctype] += len(clst)
                    cname_counts_dct[cname] += len(clst)
        # constraint count (total)
        stat["constraint_count_total"] = sum(list(ctype_counts_dct.values()))
        #   by type
        stat["constraint_count_per_type"] = dict(ctype_counts_dct)
        #   by name
        stat["constraint_count_per_name"] = dict(cname_counts_dct)
        #   per header
        hname_counts_dct = defaultdict(int)
        hname_ctype_counts_dct = defaultdict(lambda: defaultdict(int))
        for k, v in item.items():
            if not k.startswith("sampled_"):
                continue
            ctype = k[len("sampled_") :]
            for hname, constraint_dct in v.items():
                for cname, clst in constraint_dct.items():
                    if cname == "doc_structure":
                        continue
                    hname_counts_dct[hname] += len(clst)
                    hname_ctype_counts_dct[hname][ctype] += len(clst)
        stat["constraints_per_header"] = list(hname_counts_dct.values())
        stat["constraints_per_header_per_type"] = [dict(v) for v in hname_ctype_counts_dct.values()]
        stats.append(stat)

    stats_df = pd.DataFrame(stats)

    # plot stats (min, max, mean, p25, p50, p75) using logger
    log_lines: list[str] = []

    def _log_stats(df: pd.DataFrame, title: str) -> None:
        desc = df.describe(percentiles=[0.25, 0.5, 0.75]).loc[["min", "mean", "50%", "75%", "max"]].round(2)
        desc.index = ["min", "mean", "p50", "p75", "max"]
        msg = f"\n{'=' * 60}\n{title} (n={len(stats_df)})\n{'=' * 60}\n{desc.to_string()}"
        logger.info(msg)
        log_lines.append(msg)

    _log_stats(stats_df[["doc_length", "doc_depth", "leaf_count"]], "Document Size Stats")

    constraint_total_df = stats_df[["constraint_count_total"]]
    constraint_type_df = (
        pd.json_normalize(stats_df["constraint_count_per_type"]).fillna(0).sort_index(axis=1)  # sort columns by name
    )
    constraint_name_df = (
        pd.json_normalize(stats_df["constraint_count_per_name"]).fillna(0).sort_index(axis=1)  # sort columns by name
    )

    constraints_per_header_df = pd.DataFrame(
        {"constraints_per_header": pd.Series(stats_df["constraints_per_header"].explode().astype(float))}
    )
    constraints_per_header_per_type_df = (
        pd.json_normalize(stats_df["constraints_per_header_per_type"].explode()).fillna(0).sort_index(axis=1)
    )

    _log_stats(constraint_total_df, "Constraint Count Stats — Total")
    _log_stats(constraint_type_df, "Constraint Count Stats — By Type")
    _log_stats(constraint_name_df, "Constraint Count Stats — By Name")
    _log_stats(constraints_per_header_df, "Constraint Count Stats — Per Header")
    _log_stats(constraints_per_header_per_type_df, "Constraint Count Stats — By Type, Per Header")

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("\n".join(log_lines))
