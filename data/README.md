# Data

## `wiki_vital_articles_20230601_titles.jsonl`

The pipeline's document topics: the titles of the 40,148 articles listed on
[Wikipedia's Vital articles (level 5)](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5) as of
2023-06-01 (40,094 after removing duplicates).

The file is a single JSON line, `{"title": {"0": "<title>", "1": "<title>", ...}}`, in the order the pipeline reads
the topics.

Article titles come from Wikipedia, whose text is available under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
