import asyncio
import re
import yaml

from lift_bench.hierarchical_drafting.utils.inference import (
    generate_completions,
)

COMMON_KEYWORDS_PROMPT_TEMPLATE = """
I'm writing a document about the topic '{topic}'. Give me {n_words} common words that might appear in a section named '{section_name}' (hierarchy of sections is delimited by '{hierarchy_delimiter}').
Your response should be a markdown block containing only the list of keywords, for example:
```
- keyword 1
- keyword 2
- keyword 3
```
""".strip()


async def get_common_keywords(
    model_name: str, max_retries: int, document_topic: str, hierarchy_section_names: list[str], n_words: int = 5
) -> list[str]:
    """
    Generate common keywords that will appear in a document about {topic}, in section {hierarchy_section_names}.
    """
    delimiter = " / "
    section_full_name = delimiter.join(hierarchy_section_names)
    prompt = COMMON_KEYWORDS_PROMPT_TEMPLATE.format(
        topic=document_topic,
        section_name=section_full_name,
        hierarchy_delimiter=delimiter,
        n_words=n_words,
    )
    responses = await generate_completions(
        model=model_name, prompts=[prompt], max_tokens=512, temperature=0.6, n=max_retries
    )
    for completion in responses[0].choices:
        response_text = completion.message.content
        matches = re.findall(r"```[^\n]*\n(.*?)```", response_text, re.DOTALL)
        if len(matches) == 0:
            continue
        try:
            keywords = yaml.safe_load(matches[0])
            return [str(kw) for kw in keywords]
        except Exception:
            continue
    raise Exception("Failed to generate keywords")


async def main():
    result = await get_common_keywords(
        model_name="together-gpt-oss-20b",
        max_retries=3,
        document_topic="NBA",
        hierarchy_section_names=["Famous players", "Michael Jordan"],
        n_words=5,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
