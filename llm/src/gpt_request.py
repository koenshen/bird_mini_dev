#!/usr/bin/env python3
import argparse
import json
import os
import re
from openai import OpenAI
from sqlglot import parse
from sqlglot.errors import ParseError
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datetime import datetime

from prompt import generate_combined_prompts_one


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client, reasoning_effort=None, enable_thinking=None):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        time.sleep(2)
        try:

            if engine == "gpt-35-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                result = result.choices[0].text
            else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-35-turbo
                messages = [
                    {"role": "user", "content": prompt},
                ]
                extra_body = {}
                if reasoning_effort:
                    extra_body["reasoning_effort"] = reasoning_effort
                if enable_thinking is not None:
                    extra_body["enable_thinking"] = enable_thinking
                request_options = {"extra_body": extra_body} if extra_body else {}
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    **request_options,
                )
            break
        except Exception as e:
            result = "error:{}".format(e)
            print(result)
            time.sleep(4)
    return result


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    question_id_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])
        question_id_list.append(data.get("question_id", i))

    return question_list, db_path_list, knowledge_list, question_id_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for sql, _, question_id in sql_lst:
        result[str(question_id)] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


def init_client(api_key, base_url, timeout, max_retries):
    """
    Initialize an OpenAI-compatible client.
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def extract_sql_content(content):
    """Extract one SQL statement from an LLM response."""
    if not content:
        return ""

    # Thinking models commonly put their reasoning before the final answer.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]

    # A fenced SQL block provides the clearest start and end boundaries.
    fenced_sql = re.search(
        r"```(?:sql|sqlite|mysql|postgresql)?\s*([\s\S]*?)```",
        content,
        flags=re.IGNORECASE,
    )
    if fenced_sql:
        content = fenced_sql.group(1)

    # Otherwise start at the first SQL statement keyword at the start of a line.
    sql_start = re.search(
        r"(?im)^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b",
        content,
    )
    if sql_start:
        content = content[sql_start.start() :]

    content = content.strip()

    # Stop at the first semicolon outside quoted SQL values/identifiers.
    quote = None
    index = 0
    while index < len(content):
        char = content[index]
        if quote:
            if char == quote:
                # SQL escapes quote characters by doubling them.
                if index + 1 < len(content) and content[index + 1] == quote:
                    index += 2
                    continue
                quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == ";":
            return content[: index + 1].strip()
        index += 1

    return content


def post_process_response(response, db_path):
    content = (
        response if isinstance(response, str) else response.choices[0].message.content
    )
    sql = extract_sql_content(content)
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def validate_sql_syntax(sql, sql_dialect):
    """Validate one SQL statement without connecting to a database."""
    if not sql:
        return False, "empty SQL"

    try:
        statements = [
            statement for statement in parse(sql, read=sql_dialect.lower()) if statement
        ]
    except Exception as error:
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
        return False, message

    if len(statements) != 1:
        return False, f"expected one SQL statement, got {len(statements)}"
    return True, None


def worker_function(question_data):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    (
        prompt, engine, client, db_path, question, i, max_tokens, temperature,
        reasoning_effort, enable_thinking, sql_dialect, max_syntax_attempts,
        question_id,
    ) = question_data
    current_prompt = prompt
    last_sql = ""
    last_error = "not attempted"

    for attempt in range(1, max_syntax_attempts + 1):
        response = connect_gpt(
            engine,
            current_prompt,
            max_tokens,
            temperature,
            None,
            client,
            reasoning_effort,
            enable_thinking,
        )
        content = (
            response
            if isinstance(response, str)
            else response.choices[0].message.content
        )
        last_sql = extract_sql_content(content)
        valid, last_error = validate_sql_syntax(last_sql, sql_dialect)
        if valid:
            print(f"Processed {i}th question on attempt {attempt}: {question}")
            return post_process_response(last_sql, db_path), i, question_id

        print(
            f"Question {i} has invalid SQL syntax on attempt "
            f"{attempt}/{max_syntax_attempts}: {last_error}"
        )
        current_prompt = (
            f"{prompt}\n\n"
            f"The previous {sql_dialect} query has a syntax error: {last_error}\n"
            f"Previous SQL:\n{last_sql}\n"
            f"Generate one corrected {sql_dialect} statement. Return only the SQL."
        )

    print(
        f"Question {i} still has invalid SQL syntax after "
        f"{max_syntax_attempts} attempts: {last_error}"
    )
    return None


def collect_response_from_gpt(
    db_path_list,
    question_list,
    question_id_list,
    api_key,
    engine,
    sql_dialect,
    num_threads=3,
    knowledge_list=None,
    base_url=None,
    timeout=1200,
    max_retries=2,
    max_tokens=8192,
    temperature=0,
    reasoning_effort=None,
    enable_thinking=None,
    max_syntax_attempts=1,
):
    """
    Collect responses from GPT using multiple threads.
    """
    client = init_client(api_key, base_url, timeout, max_retries)

    tasks = [
        (
            generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                sql_dialect=sql_dialect,
                knowledge=knowledge_list[i] if knowledge_list else None,
            ),
            engine,
            client,
            db_path_list[i],
            question_list[i],
            i,
            max_tokens,
            temperature,
            reasoning_effort,
            enable_thinking,
            sql_dialect,
            max_syntax_attempts,
            question_id_list[i],
        )
        for i in range(len(question_list))
    ]
    responses = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task), total=len(tasks)
        ):
            response = future.result()
            if response is not None:
                responses.append(response)
    return responses


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument("--api_key", type=str, required=True)
    args_parser.add_argument("--base_url", type=str, required=True)
    args_parser.add_argument(
        "--engine", type=str, required=True, default="code-davinci-002"
    )
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--num_processes", type=int, default=3)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--temperature", type=float, default=0.0)
    args_parser.add_argument("--max_tokens", type=int, default=8192)
    args_parser.add_argument("--timeout", type=float, default=1200)
    args_parser.add_argument("--max_retries", type=int, default=2)
    args_parser.add_argument("--reasoning_effort", type=str, default=None)
    args_parser.add_argument("--enable_thinking", type=str, default=None)
    args_parser.add_argument("--max_syntax_attempts", type=int, default=1)
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list, question_id_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert (
        len(question_list)
        == len(db_path_list)
        == len(knowledge_list)
        == len(question_id_list)
    )

    if args.use_knowledge == "True":
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            question_id_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            args.num_processes,
            knowledge_list,
            args.base_url,
            args.timeout,
            args.max_retries,
            args.max_tokens,
            args.temperature,
            args.reasoning_effort,
            args.enable_thinking,
            max(1, args.max_syntax_attempts),
        )
    else:
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            question_id_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            args.num_processes,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
            max_syntax_attempts=max(1, args.max_syntax_attempts),
        )

    safe_engine_name = args.engine.replace("/", "_")
    # Include microseconds and the process ID so concurrently started runs of the
    # same model never overwrite one another's prediction file.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"
    if args.chain_of_thought == "True":
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + safe_engine_name
            + "_cot"
            + "_"
            + args.sql_dialect
            + "_"
            + run_id
            + ".json"
        )
    else:
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + safe_engine_name
            + "_"
            + args.sql_dialect
            + "_"
            + run_id
            + ".json"
        )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(f"output file: {output_name}")

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )
