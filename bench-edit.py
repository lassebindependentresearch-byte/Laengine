import ast
import json
import os
import re
from urllib.request import Request, urlopen

source = '''def cents(amount):
    return int(round(amount * 100))


def currency(value):
    return f"${value / 100:,.2f}"


def nonnegative(value):
    return max(value, 0)


def category_totals(rows):
    totals = {}
    for row in rows:
        key = row["category"]
        totals[key] = totals.get(key, 0) + row["amount"]
    return totals


def sort_categories(totals):
    return sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))


def format_table(totals):
    return "\\n".join(f"{name}: {currency(value)}" for name, value in sort_categories(totals))


def first_nonempty(values, default):
    for value in values:
        if value:
            return value
    return default


def evenly_split(total, count):
    base, remainder = divmod(total, count)
    return [base + (i < remainder) for i in range(count)]


def discount(price, percent):
    return round(price * (100 - percent) / 100)


def report(rows, title):
    totals = category_totals(rows)
    return f"{title}\\n{format_table(totals)}"
'''
copies = int(os.environ.get('LING_EDIT_COPIES', '1'))
if copies > 1:
    names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    pattern = re.compile(r'\b(?:' + '|'.join(sorted(names)) + r')\b')
    source += ''.join(
        '\n\n' + pattern.sub(lambda match: f'{match[0]}_v{copy}', source)
        for copy in range(2, copies + 1)
    )

prompt = (
    'Make exactly one change to the Python file below. In evenly_split, '
    'raise ValueError("count must be positive") if count is less than 1. '
    'Return the complete file in one Python code block. Keep every other function unchanged.\n'
    '```python\n' + source + '```'
)
request = Request(
    'http://127.0.0.1:8088/v1/chat/completions',
    data=json.dumps({
        'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
        'max_tokens': 700 * copies, 'cache_prompt': False,
    }).encode(),
    headers={'Content-Type': 'application/json'},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
answer = result['choices'][0]['message']['content']
match = re.search(r'```(?:python)?\s*\n(.*?)```', answer, re.S)
expected = source.replace('    base, remainder = divmod', '    if count < 1:\n        raise ValueError("count must be positive")\n    base, remainder = divmod', 1)
try:
    correct = bool(match) and ast.dump(ast.parse(match[1])) == ast.dump(ast.parse(expected))
except SyntaxError:
    correct = False
print(json.dumps({'mode': 'full-file edit', 'timings': result['timings'], 'correct': correct, 'content': answer}))
