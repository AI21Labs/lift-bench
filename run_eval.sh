#!/usr/bin/env bash
# Run a verifiers taskset (default: drafting_constraint_dummy) against an OpenAI-compatible endpoint.
# Results are saved to .results/<taskset>--<model>--null--<id>/traces.jsonl, one episode per line; each trace's
# `info` keeps the per-evaluation scores and ids (`agg_eval_scores`, `agg_eval_ids`).
#
# The run stays local: --no-push keeps it off the Prime Intellect platform, the subprocess runtime runs it on this
# machine, and the null harness sends the prompt as-is with no tools. Needs `uv` (the harness runs as a uv script).
#
# Usage:
#   ./run_eval.sh [extra vf-eval flags]
#
# Required env vars:
#   OPENAI_BASE_URL  OpenAI-compatible endpoint (e.g. http://localhost:8000/v1 for a local vLLM server), used for
#                    the model under test and by the LLM judge
#   OPENAI_API_KEY   its API key
#
# Optional env vars:
#   VLLM_BASE_URL  endpoint of the model under test, when it differs from the judge's (default: OPENAI_BASE_URL)
#   VLLM_API_KEY   its API key (default: OPENAI_API_KEY)
#   VLLM_MODEL     (default: Qwen/Qwen2.5-32B-Instruct)
#   NUM_EXAMPLES   (default: 1)
#   ROLLOUTS       (default: 1)
#   ENV_ID         (default: drafting_constraint_dummy; drafting_constraint needs --env.taskset.dataset-path <split>)

set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-${OPENAI_BASE_URL:-}}"
if [[ -z ${BASE_URL} ]]; then
  echo "set OPENAI_BASE_URL (or VLLM_BASE_URL) to an OpenAI-compatible endpoint" >&2
  exit 1
fi
if [[ -n ${VLLM_API_KEY:-} ]]; then
  API_KEY_VAR=VLLM_API_KEY
else
  API_KEY_VAR=OPENAI_API_KEY
fi
if [[ -z ${!API_KEY_VAR:-} ]]; then
  echo "set OPENAI_API_KEY (or VLLM_API_KEY) to the endpoint API key" >&2
  exit 1
fi
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1}"
ROLLOUTS="${ROLLOUTS:-1}"
ENV_ID="${ENV_ID:-drafting_constraint_dummy}"

PYTHONPATH=.:./environments .venv/bin/vf-eval "${ENV_ID}" \
  --model "${MODEL}" \
  --client.base-url "${BASE_URL}" \
  --client.api-key-var "${API_KEY_VAR}" \
  --env.agent.harness.id null \
  --env.agent.runtime.type subprocess \
  --no-push \
  --select.limit "${NUM_EXAMPLES}" \
  --num-rollouts "${ROLLOUTS}" \
  --output-dir .results \
  --no-rich \
  "$@"
