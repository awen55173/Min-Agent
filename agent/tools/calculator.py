"""计算器工具：基于 ast 的安全四则/幂运算求值。"""
from __future__ import annotations

import ast
import operator

from .registry import Tool, ToolError

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXP = 1000


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXP:
            raise ToolError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ToolError("unsupported expression; only numbers and + - * / // % ** ( ) are allowed")


def calculate(expression: str):
    """安全计算数学表达式，返回数值；非法表达式抛 ToolError。"""
    if len(expression) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError("invalid expression syntax") from None
    try:
        return _eval(tree)
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except OverflowError:
        raise ToolError("result too large") from None


calculator_tool = Tool(
    name="calculator",
    description="Evaluate a math expression. Supports + - * / // % ** and parentheses.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. (12+3)*4"}},
        "required": ["expression"],
    },
    func=lambda args, ctx: {"expression": args["expression"], "result": calculate(args["expression"])},
)
