#!/usr/bin/env bash
set -e

data_root='../koenshen_bird_evaluate/data_mini_dev'
eval_path="${data_root}/dev.json"
db_root_path="${data_root}/dev_databases/"
use_knowledge='True'
mode='mini_dev'
cot='True'

base_url='http://10.210.9.14:50011/v1'
api_key='empty'
engine='flash-lite-v36-fp8-step3k-dpo-mtp1'

temperature=1
max_tokens=8192
timeout=1200
max_retries=2
num_threads=10
max_syntax_attempts=20
sql_dialect='SQLite'
output_path='./llm/exp_result/tokenhub_output_kg/'

echo "base_url: ${base_url}"
echo "model: ${engine}"
echo "api_key: $([[ -n "${api_key}" ]] && echo '<set>' || echo '<empty>')"
echo "eval_path: ${eval_path}"
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
