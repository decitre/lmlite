# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Tool dispatch and the stop conditions of ToolRunner, with a model that replies from a script."""

import asyncio
import json

import pytest

from lmlite.tools import Tool, ToolRunner, model_tool


class FakeModel:
    def __init__(self, *replies, delay=0):
        self.replies, self.delay, self.seen, self.cancelled = list(replies), delay, [], False

    async def chat(self, messages):
        self.seen.append(list(messages))
        await asyncio.sleep(self.delay)
        return self.replies.pop(0)

    async def cancel(self):
        self.cancelled = True


def call(name, **arguments):
    return json.dumps({"tool": name, "arguments": arguments})


def answer(text):
    return json.dumps({"answer": text})


@pytest.fixture
def calls():
    return []


@pytest.fixture
def tools(calls):
    def get_weather(city, unit="celsius"):
        calls.append((city, unit))
        return f"18 degrees in {city}"

    async def search(query):
        calls.append(query)
        return {"hits": [query]}

    return [
        Tool(
            "get_weather",
            "Weather of a city.",
            get_weather,
            {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
                "required": ["city"],
            },
        ),
        model_tool(search, "search", "Search the wiki."),
    ]


async def test_a_tool_call_returns_its_observation_to_the_model_and_the_model_answers(tools, calls):
    model = FakeModel(call("get_weather", city="Lisbon"), answer("Mild: 18 degrees."))
    result = await ToolRunner(model, tools).run("Weather in Lisbon?")

    assert (result.answer, result.stopped) == ("Mild: 18 degrees.", "answer")
    assert calls == [("Lisbon", "celsius")]
    assert [(s["outcome"], s.get("tool")) for s in result.steps] == [("ok", "get_weather"), ("answer", None)]
    assert "18 degrees in Lisbon" in model.seen[1][-1]["content"]
    assert "get_weather" in model.seen[0][0]["content"]  # the system message lists the tools


async def test_an_async_tool_made_with_model_tool_gets_the_string_and_its_result_is_json(tools, calls):
    model = FakeModel(call("search", text="cats"), answer("done"))
    result = await ToolRunner(model, tools).run("q")

    assert calls == ["cats"] and result.steps[0]["detail"] == '{"hits": ["cats"]}'


@pytest.mark.parametrize(
    "reply, reason",
    [
        (call("launch_missiles"), "unknown tool"),
        (call("get_weather"), "missing 'city'"),
        (call("get_weather", city=5), "must be of type string"),
        (call("get_weather", city="Rome", unit="kelvin"), "must be one of"),
        (call("get_weather", city="Rome", extra=1), "unexpected property 'extra'"),
        (json.dumps({"tool": "get_weather", "arguments": ["Rome"]}), "must be of type object"),
        ("I think it is sunny.", "not a JSON object"),
        ("[1, 2]", "not a JSON object"),
        (json.dumps({"answer": 5}), "answer must be a string"),
        (json.dumps({"tool": ["get_weather"], "arguments": {}}), "unknown tool"),
        (json.dumps({"tool": {"name": "search"}}), "unknown tool"),
    ],
)
async def test_an_invalid_reply_never_runs_a_tool_and_the_model_may_correct_it(tools, calls, reply, reason):
    model = FakeModel(reply, answer("sorry"))
    result = await ToolRunner(model, tools).run("q")

    assert calls == []
    assert result.steps[0]["outcome"] == "invalid" and reason in result.steps[0]["detail"]
    assert "Refused" in model.seen[1][-1]["content"]
    assert result.answer == "sorry"


async def test_json_wrapped_in_prose_or_a_code_fence_is_read(tools, calls):
    model = FakeModel("Sure!\n```json\n" + call("get_weather", city="Oslo") + "\n```", answer("ok"))
    await ToolRunner(model, tools).run("q")

    assert calls == [("Oslo", "celsius")]


async def test_a_number_is_not_a_boolean_and_a_boolean_is_not_a_number():
    tool = Tool("t", "d", lambda **a: "ok", {"type": "object", "properties": {"n": {"type": "integer"}}})
    model = FakeModel(call("t", n=True), answer("a"))
    result = await ToolRunner(model, [tool]).run("q")

    assert result.steps[0]["outcome"] == "invalid"


async def test_a_failing_tool_is_an_observation_and_the_run_goes_on():
    def broken():
        raise RuntimeError("camera busy")

    model = FakeModel(call("broken"), answer("I could not."))
    result = await ToolRunner(model, [Tool("broken", "d", broken)]).run("q")

    assert result.steps[0]["outcome"] == "error" and "RuntimeError: camera busy" in result.steps[0]["detail"]
    assert "camera busy" in model.seen[1][-1]["content"] and result.answer == "I could not."


async def test_a_slow_tool_times_out_as_an_error():
    async def slow():
        await asyncio.sleep(5)

    model = FakeModel(call("slow"), answer("gave up"))
    result = await ToolRunner(model, [Tool("slow", "d", slow)], tool_timeout=0.01).run("q")

    assert result.steps[0]["outcome"] == "error" and "TimeoutError" in result.steps[0]["detail"]


async def test_the_step_limit_stops_a_model_that_keeps_calling_tools(tools, calls):
    model = FakeModel(*[call("get_weather", city="A")] * 5)
    result = await ToolRunner(model, tools, max_steps=3).run("q")

    assert (result.answer, result.stopped, len(result.steps), len(calls)) == (None, "step_limit", 3, 3)


async def test_invalid_replies_count_against_the_step_limit(tools):
    model = FakeModel("no", "no", "no")
    result = await ToolRunner(model, tools, max_steps=2).run("q")

    assert result.stopped == "step_limit" and [s["outcome"] for s in result.steps] == ["invalid", "invalid"]


async def test_the_timeout_ends_the_run_with_the_steps_so_far_and_stops_the_generation(tools):
    model = FakeModel(call("get_weather", city="A"), call("get_weather", city="B"), delay=0.2)
    result = await ToolRunner(model, tools, timeout=0.3).run("q")

    assert result.stopped == "timeout" and result.answer is None
    assert [s["outcome"] for s in result.steps] == ["ok"]  # the second reply never arrived
    assert model.cancelled


async def test_cancelling_the_task_stops_the_generation_and_raises(tools):
    model = FakeModel(answer("never"), delay=5)
    task = asyncio.ensure_future(ToolRunner(model, tools).run("q"))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.cancelled


async def test_a_long_observation_is_cut(tools):
    tool = Tool("big", "d", lambda: "x" * 100)
    model = FakeModel(call("big"), answer("a"))
    result = await ToolRunner(model, [tool], max_observation=10).run("q")

    assert result.steps[0]["detail"] == "x" * 10 + " ...[truncated]"


def test_bad_setup_is_refused(tools):
    with pytest.raises(ValueError, match="distinct"):
        ToolRunner(FakeModel(), [tools[0], tools[0]])
    with pytest.raises(ValueError, match="max_steps"):
        ToolRunner(FakeModel(), tools, max_steps=0)


def test_a_schema_the_validator_cannot_apply_is_refused_at_setup():
    for bad in (
        {"type": "null"},
        {"type": ["string", "null"]},
        {"type": "object", "properties": {"x": {"type": "null"}}},
    ):
        with pytest.raises(ValueError, match="Unsupported schema type"):
            ToolRunner(FakeModel(), [Tool("t", "d", lambda **a: "", bad)])


async def test_an_object_argument_without_properties_takes_any_keys():
    seen = []
    schema = {"type": "object", "properties": {"filters": {"type": "object"}}}
    model = FakeModel(call("t", filters={"a": 1}), answer("ok"))
    await ToolRunner(model, [Tool("t", "d", lambda **a: seen.append(a) or "ok", schema)]).run("q")

    assert seen == [{"filters": {"a": 1}}]


async def test_tools_may_be_a_generator(tools):
    result = await ToolRunner(FakeModel(answer("hi")), (tool for tool in tools)).run("q")

    assert result.answer == "hi"


async def test_a_timeout_that_is_not_the_runs_is_an_error():
    class Stalled:
        async def chat(self, messages):
            raise asyncio.TimeoutError("connection stalled")

    with pytest.raises(asyncio.TimeoutError):
        await ToolRunner(Stalled(), []).run("q")


async def test_plain_text_after_a_tool_result_is_the_answer(tools, calls):
    model = FakeModel(call("get_weather", city="Lisbon"), "It is 18 degrees in Lisbon.")
    result = await ToolRunner(model, tools).run("Weather in Lisbon?")

    assert (result.answer, result.stopped) == ("It is 18 degrees in Lisbon.", "answer")
    assert result.steps[-1]["plain_text"] is True and calls == [("Lisbon", "celsius")]


async def test_a_tool_error_also_lets_the_model_answer_in_plain_text():
    def broken():
        raise RuntimeError("busy")

    result = await ToolRunner(FakeModel(call("broken"), "Sorry, it was busy."), [Tool("broken", "d", broken)]).run("q")

    assert result.answer == "Sorry, it was busy."


async def test_plain_text_is_still_refused_before_any_tool_ran(tools):
    model = FakeModel("It is sunny.", "Still no JSON.", answer("ok"))
    result = await ToolRunner(model, tools).run("q")

    assert [s["outcome"] for s in result.steps] == ["invalid", "invalid", "answer"] and result.answer == "ok"
    assert "plain_text" not in result.steps[-1]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        '{"error": {"message": "Unknown tool"}}',
        call("launch_missiles"),
        '{"tool": "get_weather", "arguments": {"city": "Rome", ',  # cut off at the token limit
        '{"tool": "get_weather", "arguments": {"city": "Rome",}}',  # a trailing comma
        "See {city} for details",  # braces: it looks like JSON, so it is not taken as the answer
        "```json\n" + call("get_weather", city="Rome")[:-3] + "\n```",  # a broken call in a code fence
    ],
)
async def test_after_a_tool_result_only_prose_counts_as_an_answer(tools, reply):
    model = FakeModel(call("get_weather", city="Rome"), reply, answer("fine"))
    result = await ToolRunner(model, tools).run("q")

    assert result.steps[1]["outcome"] == "invalid" and result.answer == "fine"
