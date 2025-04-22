#!/usr/bin/env python3
import re
import ast
import json
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

# --- Gemini API config --- #
GEMINI_API_KEY = '...'
GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
GEMINI_MODEL_ID = 'gemini-2.0-flash'


# --- structured output model --- #
class ComparisonResult(BaseModel):
    name: str
    equivalent: bool
    differences: str


# --- OpenAI client pointed at Gemini --- #
client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def normalize(name: str) -> str:
    return name.replace('_', '').lower()


def extract_python_functions(source: str):
    lines = source.splitlines()
    tree = ast.parse(source)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min((d.lineno for d in node.decorator_list), default=node.lineno)
            end = getattr(node, 'end_lineno', None)
            if end is None:
                base_indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
                end = start
                for i in range(start, len(lines)):
                    if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= base_indent:
                        break
                    end = i + 1
            code = '\n'.join(lines[start - 1 : end])
            orig = node.name
            funcs[normalize(orig)] = (orig, code)
    return funcs


def extract_cpp_functions(source: str):
    lines = source.splitlines()
    sig = re.compile(r'^\s*(?:[\w:<>\*&\s]+?)\s+([A-Za-z_]\w*(?:::\w+)?)\s*\([^;]*\)\s*(?:const\s*)?\{')
    funcs = {}
    i = 0
    while i < len(lines):
        m = sig.match(lines[i])
        if not m:
            i += 1
            continue
        orig = m.group(1)
        norm = normalize(orig)
        brace = 0
        start = i
        for j in range(i, len(lines)):
            brace += lines[j].count('{') - lines[j].count('}')
            if brace == 0:
                block = '\n'.join(lines[start : j + 1])
                funcs[norm] = (orig, block)
                i = j + 1
                break
        else:
            funcs[norm] = (orig, '\n'.join(lines[start:]))
            break
    return funcs


def compare_logic(cpp_code: str, py_code: str, name: str) -> ComparisonResult:
    system_prompt = (
        'You are an expert that compares two implementations of the same function—'
        'one in C++ and one in Python—and determines whether they implement equivalent logic.\n\n'
        'OUTPUT **ONLY** a JSON object matching this schema:\n'
        '  - name: string\n'
        '  - equivalent: boolean\n'
        '  - differences: string (empty if equivalent, otherwise a brief bullet‑list)\n'
    )
    user_prompt = f'Function: "{name}"\n\n<<<C++>>>\n' + cpp_code + '\n\n<<<PYTHON>>>\n' + py_code

    completion = client.beta.chat.completions.parse(
        model=GEMINI_MODEL_ID,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.0,
        response_format=ComparisonResult,
    )
    return completion.choices[0].message.parsed  # type: ignore


def main(py_path, cpp_path):
    py_src = Path(py_path).read_text(encoding='utf-8')
    cpp_src = Path(cpp_path).read_text(encoding='utf-8')

    py_funcs = extract_python_functions(py_src)
    cpp_funcs = extract_cpp_functions(cpp_src)

    common = sorted(set(py_funcs) & set(cpp_funcs))
    output = ''
    for norm in tqdm(common, desc='Comparing'):
        _, py_code = py_funcs[norm]
        _, cpp_code = cpp_funcs[norm]
        try:
            comp = compare_logic(cpp_code, py_code, norm)
            data = comp.model_dump()
        except Exception as e:
            data = {'name': norm, 'equivalent': False, 'differences': f'<error: {e}>'}

        time.sleep(10)  # rate limit

        if not data['equivalent']:
            output += json.dumps(data, ensure_ascii=False) + '\n'
            print(f"Function '{data['name']}' is NOT equivalent:\n{data['differences']}")

    print(output, end='')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compare logic of matching Python/C++ functions via Gemini structured output'
    )
    p.add_argument('python_file', help='Path to .py source')
    p.add_argument('cpp_file', help='Path to .cpp/.h source')
    args = p.parse_args()
    main(args.python_file, args.cpp_file)
