import ast
import difflib
from dataclasses import dataclass

from .data import digest


@dataclass(frozen=True)
class Candidate:
    source: str
    operation: str
    line: int
    sha256: str


def replace_node(source: str, node: ast.AST, replacement: str) -> str:
    encoded = source.encode("utf-8")
    lines = encoded.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    return (encoded[:start] + replacement.encode("utf-8") + encoded[end:]).decode("utf-8")


def propose(source: str, budget: int = 12) -> list[Candidate]:
    if not 1 <= budget <= 128:
        raise ValueError("Candidate budget must be between 1 and 128")
    tree = ast.parse(source)
    proposals = []
    comparisons = {ast.Lt: "<=", ast.LtE: "<", ast.Gt: ">=", ast.GtE: ">", ast.Eq: "!=", ast.NotEq: "=="}
    binary = {ast.Add: "-", ast.Sub: "+", ast.Mult: "+", ast.FloorDiv: "%", ast.Mod: "//"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in comparisons:
            expression = f"({ast.unparse(node.left)} {comparisons[type(node.ops[0])]} {ast.unparse(node.comparators[0])})"
            proposals.append((node, expression, "comparison"))
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            proposals.append((node, f"({ast.unparse(node.left)} {binary[type(node.op)]} {ast.unparse(node.right)})", "arithmetic"))
        elif isinstance(node, ast.Call) and len(node.args) == 2 and not node.keywords and not any(isinstance(x, ast.Starred) for x in node.args):
            proposals.append((node, f"{ast.unparse(node.func)}({ast.unparse(node.args[1])}, {ast.unparse(node.args[0])})", "swap_arguments"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1):
            proposals.append((node, str(1 - node.value), "boundary_constant"))
    result = []
    seen = {digest(source.encode())}
    for node, replacement, operation in sorted(proposals, key=lambda x: (x[0].lineno, x[0].col_offset, x[2])):
        changed = replace_node(source, node, replacement)
        try:
            ast.parse(changed)
        except SyntaxError:
            continue
        sha = digest(changed.encode())
        if sha not in seen:
            result.append(Candidate(changed, operation, node.lineno, sha))
            seen.add(sha)
        if len(result) == budget:
            break
    return result


def patch(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))
