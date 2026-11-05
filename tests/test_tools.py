import pytest

from agent.config import Settings
from agent.tools import ToolContext, ToolRegistry, TodoStore, build_registry
from agent.tools.calculator import calculate
from agent.tools.registry import Tool, ToolError

import json

A1 = ToolContext("alice", "s1")
A2 = ToolContext("alice", "s2")
B1 = ToolContext("bob", "s1")

WTTR = {"current_condition": [{"temp_C": "20", "FeelsLikeC": "19", "humidity": "50",
                               "weatherDesc": [{"value": "Sunny"}]}]}


def run(reg, name, args, ctx=A1):
    r = reg.execute(name, args, ctx)
    return r.ok, (json.loads(r.content) if r.ok else r.content)


@pytest.fixture
def reg():
    return build_registry(
        tavily_api_key="key",
        http_get=lambda url: WTTR,
        http_post=lambda url, body, key: {"results": [
            {"title": "T", "url": "http://u", "content": "x" * 500, "extra": 1}]},
    )


# ---- registry ----
def test_registry_registers_all_four(reg):
    assert set(reg.names()) == {"calculator", "todo", "search", "weather"}
    for s in reg.schemas():
        assert s["name"] and s["description"] and s["parameters"]["type"] == "object"


def test_duplicate_register_rejected():
    r = ToolRegistry()
    t = Tool("x", "d", {"type": "object", "properties": {}}, lambda a, c: "ok")
    r.register(t)
    with pytest.raises(ValueError):
        r.register(t)


def test_decorator_register():
    r = ToolRegistry()

    @r.tool("echo", "d", {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]})
    def echo(args, ctx):
        return args["t"]

    assert r.execute("echo", {"t": "hi"}).content == "hi"


def test_unknown_tool(reg):
    ok, msg = run(reg, "nope", {})
    assert not ok and "calculator" in msg


@pytest.mark.parametrize("args,frag", [
    ({}, "missing"), ({"expression": 1}, "must be string"),
    ({"expression": "1", "x": 1}, "unknown"), ("1+1", "JSON object"),
])
def test_arg_validation(reg, args, frag):
    ok, msg = run(reg, "calculator", args)
    assert not ok and frag in msg


def test_bool_not_integer(reg):
    ok, msg = run(reg, "todo", {"action": "complete", "id": True})
    assert not ok and "integer" in msg


def test_enum_validation(reg):
    ok, msg = run(reg, "todo", {"action": "explode"})
    assert not ok and "one of" in msg


def test_unexpected_exception_is_contained():
    r = ToolRegistry()
    r.register(Tool("bad", "d", {"type": "object", "properties": {}}, lambda a, c: 1 / 0))
    res = r.execute("bad", {})
    assert not res.ok and "ZeroDivisionError" in res.content


# ---- calculator ----
@pytest.mark.parametrize("expr,val", [("1+2*3", 7), ("(12+3)*4", 60), ("2**10", 1024), ("-5+2", -3),
                                       ("7//2", 3), ("7%4", 3), ("1/4", 0.25)])
def test_calc_ok(expr, val):
    assert calculate(expr) == val


@pytest.mark.parametrize("expr", ["__import__('os')", "1+", "abs(1)", "a+1", "1/0", "2**100000", "'a'*3",
                                  "(1).real", "1;2", "x" * 300,
                                  "(-1)**0.5", "(10**100)**10", "1e309", "((9**9)**9)**9", "10**101", "1e200*1e200",
                                  "2**1001", "9**9**9", "0**-1"])
def test_calc_rejects(expr):
    with pytest.raises(ToolError):
        calculate(expr)


# ---- todo ----
def test_todo_crud(reg):
    ok, r = run(reg, "todo", {"action": "add", "text": "write report"})
    assert ok and r["created"] and r["item"]["id"] == 1
    run(reg, "todo", {"action": "add", "text": "buy milk"})
    ok, r = run(reg, "todo", {"action": "list"})
    assert [i["text"] for i in r["items"]] == ["write report", "buy milk"]
    ok, r = run(reg, "todo", {"action": "complete", "id": 1})
    assert r["complete"]["done"] is True
    _, r = run(reg, "todo", {"action": "list", "status": "pending"})
    assert [i["id"] for i in r["items"]] == [2]
    _, r = run(reg, "todo", {"action": "list", "status": "done"})
    assert [i["id"] for i in r["items"]] == [1]
    ok, r = run(reg, "todo", {"action": "delete", "id": 2})
    assert ok
    _, r = run(reg, "todo", {"action": "list"})
    assert [i["id"] for i in r["items"]] == [1]


def test_todo_ids_not_reused_after_delete(reg):
    run(reg, "todo", {"action": "add", "text": "a"})
    run(reg, "todo", {"action": "delete", "id": 1})
    _, r = run(reg, "todo", {"action": "add", "text": "b"})
    assert r["item"]["id"] == 2


def test_todo_errors(reg):
    assert not run(reg, "todo", {"action": "add"})[0]
    assert not run(reg, "todo", {"action": "add", "text": "  "})[0]
    assert not run(reg, "todo", {"action": "complete"})[0]
    ok, msg = run(reg, "todo", {"action": "delete", "id": 99})
    assert not ok and "not found" in msg


def test_todo_add_idempotent_within_same_turn(reg):
    ctx = ToolContext("alice", "s1", turn_id=1)
    run(reg, "todo", {"action": "add", "text": "same"}, ctx)
    _, r = run(reg, "todo", {"action": "add", "text": "same"}, ctx)
    assert r["created"] is False and r["item"]["id"] == 1
    _, r = run(reg, "todo", {"action": "list"}, ctx)
    assert len(r["items"]) == 1
    assert "turn" not in r["items"][0]


def test_todo_add_again_in_later_turn_creates_new(reg):
    run(reg, "todo", {"action": "add", "text": "same"}, ToolContext("alice", "s1", turn_id=1))
    _, r = run(reg, "todo", {"action": "add", "text": "same"}, ToolContext("alice", "s1", turn_id=2))
    assert r["created"] is True and r["item"]["id"] == 2


def test_todo_no_dedupe_without_turn_id(reg):
    run(reg, "todo", {"action": "add", "text": "same"})
    _, r = run(reg, "todo", {"action": "add", "text": "same"})
    assert r["created"] is True


def test_todo_isolation_between_sessions_and_users(reg):
    run(reg, "todo", {"action": "add", "text": "weather"}, A1)
    run(reg, "todo", {"action": "add", "text": "report"}, A2)
    run(reg, "todo", {"action": "add", "text": "other"}, B1)
    for ctx, text in ((A1, "weather"), (A2, "report"), (B1, "other")):
        _, r = run(reg, "todo", {"action": "list"}, ctx)
        assert [i["text"] for i in r["items"]] == [text]
        assert r["items"][0]["id"] == 1
    # 操作 A2 不影响 A1
    run(reg, "todo", {"action": "complete", "id": 1}, A2)
    _, r = run(reg, "todo", {"action": "list"}, A1)
    assert r["items"][0]["done"] is False
    assert not run(reg, "todo", {"action": "delete", "id": 1}, ToolContext("carol", "s"))[0]


def test_shared_store_across_registries():
    store = TodoStore()
    r1, r2 = build_registry(todo_store=store), build_registry(todo_store=store)
    run(r1, "todo", {"action": "add", "text": "x"})
    assert len(run(r2, "todo", {"action": "list"})[1]["items"]) == 1


# ---- weather ----
def test_weather_ok(reg):
    ok, r = run(reg, "weather", {"city": "Beijing"})
    assert ok and r == {"city": "Beijing", "temp_c": "20", "feels_like_c": "19",
                        "humidity": "50", "description": "Sunny"}


def test_weather_url_and_errors():
    seen = []
    reg = build_registry(http_get=lambda u: seen.append(u) or {})
    ok, msg = run(reg, "weather", {"city": "New York"})
    assert not ok and "no weather data" in msg
    assert seen == ["https://wttr.in/New%20York?format=j1"]
    assert not run(reg, "weather", {"city": " "})[0]


def test_weather_service_down():
    def boom(url):
        raise ToolError("weather service unavailable: X")

    ok, msg = run(build_registry(http_get=boom), "weather", {"city": "x"})
    assert not ok and "unavailable" in msg


# ---- search ----
def test_search_ok_and_truncates(reg):
    ok, r = run(reg, "search", {"query": "python"})
    assert ok and r["results"] == [{"title": "T", "url": "http://u", "snippet": "x" * 300}]


def test_search_passes_params():
    seen = {}
    reg = build_registry("k", http_post=lambda u, b, k: seen.update(b=b, k=k) or {"results": []})
    run(reg, "search", {"query": " q ", "max_results": 5})
    assert seen == {"b": {"query": "q", "max_results": 5}, "k": "k"}


def test_search_errors(reg):
    assert not run(reg, "search", {"query": " "})[0]
    assert not run(reg, "search", {"query": "q", "max_results": 0})[0]
    ok, msg = run(build_registry(None), "search", {"query": "q"})
    assert not ok and "TAVILY_API_KEY" in msg
    bad = build_registry("k", http_post=lambda u, b, k: {"oops": 1})
    assert not run(bad, "search", {"query": "q"})[0]


# ---- live (默认跳过) ----
@pytest.mark.live
def test_live_weather():
    ok, r = run(build_registry(), "weather", {"city": "Beijing"})
    assert ok and "temp_c" in r


@pytest.mark.live
def test_live_search():
    ok, r = run(build_registry(Settings.from_env().tavily_api_key), "search", {"query": "python language", "max_results": 2})
    assert ok and r["results"]


def test_result_serialization_failure_is_contained():
    r = ToolRegistry()
    r.register(Tool("nan", "d", {"type": "object", "properties": {}}, lambda a, c: {"v": float("nan")}))
    r.register(Tool("cplx", "d", {"type": "object", "properties": {}}, lambda a, c: {"v": 1j}))
    r.register(Tool("big", "d", {"type": "object", "properties": {}}, lambda a, c: {"v": 10 ** 5000}))
    for name in ("nan", "cplx", "big"):
        res = r.execute(name, {})
        assert not res.ok and "serializable" in res.content


def test_calculator_tool_end_to_end_rejections(reg):
    for expr in ("(-1)**0.5", "(10**100)**10", "1e309"):
        ok, msg = run(reg, "calculator", {"expression": expr})
        assert not ok and isinstance(msg, str)


def test_calculator_boundary_values_still_work():
    assert calculate("10**100") == 10 ** 100
    assert calculate("2**300") == 2 ** 300
    assert calculate("1e100") == 1e100
