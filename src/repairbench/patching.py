import ast
import re

from .data import Task


def signatures(source: str):
    tree = ast.parse(source)
    return {node.name: ast.dump(node.args, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def apply_response(task: Task, response: str) -> tuple[str, dict[str, bytes]]:
    if len(response.encode("utf-8")) > 65536:
        raise ValueError("Response exceeds 64 KiB")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.DOTALL)
    if len(blocks) != 1 or response.count("```") != 2:
        raise ValueError("Expected exactly one fenced Python module")
    source = blocks[0].strip() + "\n"
    try:
        before, after = signatures(task.source), signatures(source)
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ValueError("Patch is not a parseable Python module") from exc
    if not before or any(after.get(name) != signature for name, signature in before.items()):
        raise ValueError("Patch changes a required function signature")
    if ast.dump(ast.parse(source)) == ast.dump(ast.parse(task.source)):
        raise ValueError("Patch does not change program behavior structurally")
    payload = dict(task.files)
    target = f"python_programs/{task.name}.py"
    if target not in payload:
        raise ValueError("Missing target program")
    payload[target] = source.encode("utf-8")
    return source, payload
