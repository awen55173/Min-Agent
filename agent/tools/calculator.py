"""计算器工具：基于 ast 的安全四则/幂运算求值，限制数值范围。"""
from __future__ import annotations

import ast
import math
import operator

from .registry import Tool, ToolError

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXP = 1000  # 单次幂运算的最大指数
_MAX_ABS = 1e100  # 常量、每个中间结果及最终结果的绝对值上限
_MAX_POW_BITS = 333  # 整数幂结果的预估最大二进制位数（约 1e100），计算前拦截


def _check(value):
    """校验数值为有限实数且在范围内，否则抛 ToolError；返回原值。"""
    if isinstance(value, complex):
        raise ToolError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("number is not finite")
    if abs(value) > _MAX_ABS:
        raise ToolError(f"number out of supported range (|x| <= {_MAX_ABS:g})")
    return value


def _eval(node):
    """递归求值白名单语法节点，每一步都做范围检查。"""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _check(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXP:
                raise ToolError("exponent too large")
            if isinstance(left, int) and isinstance(right, int) and right > 0 and abs(left) > 1 \
                    and math.log2(abs(left)) * right > _MAX_POW_BITS:
                raise ToolError("result too large")
        return _check(_BIN[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_eval(node.operand)))
    raise ToolError("unsupported expression; only numbers and + - * / // % ** ( ) are allowed")


def calculate(expression: str):
    """安全计算数学表达式，返回有限实数；非法或超范围的表达式抛 ToolError。"""
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
