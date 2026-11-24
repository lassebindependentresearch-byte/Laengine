import ast
import json
import os
from pathlib import Path
import random
import re
import time
from urllib.request import Request, urlopen

base = Path(__file__).parent
server = os.getenv('LING_TEST_URL', 'http://127.0.0.1:8088')
target = int(os.getenv('LING_DEPTH', '265000'))
task = os.getenv('LING_LONG_TASK', 'edit')


def post(path, data, timeout=40):
    req = Request(server + path, data=json.dumps(data).encode(),
                  headers={'Content-Type': 'application/json'})
    return urlopen(req, timeout=timeout)


tree = ast.parse((base / 'bench-edit.py').read_text())
source = next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == 'source')
names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
pattern = re.compile(r'\b(?:' + '|'.join(sorted(names)) + r')\b')
source += ''.join(
    '\n\n' + pattern.sub(lambda match: f'{match[0]}_v{copy}', source)
    for copy in range(2, int(os.getenv('LING_EDIT_COPIES', '3')) + 1)
)
expected = source.replace(
    '    base, remainder = divmod',
    '    if count < 1:\n        raise ValueError("count must be positive")\n    base, remainder = divmod',
    1,
)
seal = 'VELVET-BIRCH-48371'
rng = random.Random(312)
records = [
    f'Archive record {i:05d}: parcel {rng.randrange(100000,999999)}, zone {rng.randrange(10,99)}, '
    f'units {rng.randrange(5,500)}, route {rng.choice(("ridge", "shore", "delta", "meadow", "pine"))}.\n'
    for i in range(target // 12 + 200)
]
prefix = f'Archive manifest. The archive seal is {seal}.\n'
instruction = (
    '\nEnd of archive. Edit the Python file below. In evenly_split, raise '
    'ValueError("count must be positive") if count is less than 1. Keep all other functions unchanged. '
    'Return the complete file in one Python code block. After it, give the exact archive seal '
    'shown in the manifest at the start.\n```python\n' + source + '```'
)


def messages(n):
    return [{'role': 'user', 'content': prefix + ''.join(records[:n]) + instruction}]


def token_count(n):
    with post('/apply-template', {'messages': messages(n)}) as response:
        prompt = json.load(response)['prompt']
    with post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True}) as response:
        return len(json.load(response)['tokens'])


lo, hi = 0, len(records)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if token_count(mid) <= target:
        lo = mid
    else:
        hi = mid - 1
planned = token_count(lo)
assert target - 100 <= planned <= target, planned
if task == 'chat':
    instruction += (
        '\n\nFinal instruction: skip the earlier Python edit. Begin with the archive seal '
        'from the manifest. In about 300 words, explain a design for importing the parcel '
        'records into a database. Cover idempotency, validation, audit logs, and how to '
        'recover if the import fails halfway. Use concrete examples.'
    )
    planned = token_count(lo)
    assert planned < 265500, planned
print(json.dumps({'planned_prompt_tokens': planned, 'archive_records': lo}), flush=True)
start = time.monotonic()
ttft = None
chunks = []
usage = timings = None
finish_reason = None
with post('/v1/chat/completions', {
    'messages': messages(lo), 'temperature': 0, 'max_tokens': 650 if task == 'chat' else 1140,
    'stream': True, 'stream_options': {'include_usage': True},
    'cache_prompt': os.getenv('LING_CACHE_PROMPT') == '1',
}, timeout=14400) as response:
    for raw in response:
        if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
            continue
        chunk = json.loads(raw[6:])
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {}).get('content')
            if delta:
                if ttft is None:
                    ttft = time.monotonic() - start
                    print(json.dumps({'time_to_first_token_seconds': ttft}), flush=True)
                chunks.append(delta)
            finish_reason = choice.get('finish_reason') or finish_reason
        usage = chunk.get('usage') or usage
        timings = chunk.get('timings') or timings
answer = ''.join(chunks)
match = re.search(r'```(?:python)?\s*\n(.*?)```', answer, re.S)
try:
    edit_correct = bool(match) and ast.dump(ast.parse(match[1])) == ast.dump(ast.parse(expected))
except SyntaxError:
    edit_correct = False
print(json.dumps({
    'prompt_tokens_planned': planned,
    'usage': usage,
    'timings': timings,
    'wall_seconds': time.monotonic() - start,
    'finish_reason': finish_reason,
    'edit_correct': edit_correct,
    'seal_correct': seal in answer,
    'answer': answer,
}), flush=True)
