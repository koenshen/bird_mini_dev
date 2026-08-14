#!/usr/bin/env bash
set -e

# bash llm/run/run_tokenhub_sense_260804_junhao.sh --prompt_jsonl llm/run/dev_prompt.jsonl --output_file ./llm/exp_result/tokenhub_output_kg/predict_dev_sjxb_Qwen3_6_27B_bird_v12_cot_SQLite-260812-1.json

data_root='../koenshen_bird_evaluate/data_dev'
eval_path="${data_root}/dev.json"
prompt_jsonl="${PROMPT_JSONL:-}"
db_root_path="${data_root}/dev_databases/"
use_knowledge='True'
mode='dev'
cot='True'

# 部署的模型名就是Qwen3.6-27B，部署老师限制的
base_url='http://10.142.85.18:31181/v1'
api_key='empty'
engine='Qwen3.6-27B'

temperature=1.0
max_tokens=32768
timeout=1200
max_retries=2
num_threads=50
max_syntax_attempts=20
sql_dialect='SQLite'
output_path='./llm/exp_result/tokenhub_output_kg/'

echo "base_url: ${base_url}"
echo "model: ${engine}"
echo "api_key: $([[ -n "${api_key}" ]] && echo '<set>' || echo '<empty>')"
echo "eval_path: ${eval_path}"
echo "prompt_jsonl: ${prompt_jsonl:-<not set; using eval_path>}"
echo "db_root_path: ${db_root_path}"
echo "output_path: ${output_path}"
echo "temperature: ${temperature}"
echo "max_tokens: ${max_tokens}"
echo "timeout: ${timeout}"
echo "max_retries: ${max_retries}"
echo "num_threads: ${num_threads}"
echo "max_syntax_attempts: ${max_syntax_attempts}"
echo "sql_dialect: ${sql_dialect}"
echo "use_knowledge: ${use_knowledge}"
echo "chain_of_thought: ${cot}"
echo "extra arguments: ${*:-<none>}"

uv run --with-requirements ./requirements.txt \
  python -u ./llm/src/gpt_request.py \
  --base_url "${base_url}" \
  --api_key "${api_key}" \
  --engine "${engine}" \
  --eval_path "${eval_path}" \
  --prompt_jsonl "${prompt_jsonl}" \
  --db_root_path "${db_root_path}" \
  --data_output_path "${output_path}" \
  --mode "${mode}" \
  --use_knowledge "${use_knowledge}" \
  --chain_of_thought "${cot}" \
  --num_processes "${num_threads}" \
  --sql_dialect "${sql_dialect}" \
  --temperature "${temperature}" \
  --max_tokens "${max_tokens}" \
  --timeout "${timeout}" \
  --max_retries "${max_retries}" \
  --max_syntax_attempts "${max_syntax_attempts}" \
  "$@"
