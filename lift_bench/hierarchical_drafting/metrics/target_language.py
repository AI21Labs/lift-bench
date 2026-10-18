import langid

from lift_bench.hierarchical_drafting.metrics.registry import register_metric


@register_metric("target_language")
async def target_language(completion, language_code: str) -> float:
    text = completion[-1]["content"]
    if len(text) <= 1:
        return 1.0
    detected_lang = langid.classify(text)[0]
    return 1.0 if detected_lang == language_code else 0.0
