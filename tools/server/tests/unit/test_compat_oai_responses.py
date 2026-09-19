import pytest
from pathlib import Path
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/responses/input_tokens"])
def test_responses_function_call_output_content_array(endpoint):
    server.jinja = True
    server.start()
    res = server.make_request("POST", endpoint, data={
        "input": [
            {"role": "user", "content": "Describe the image"},
            {"type": "function_call", "call_id": "call_image", "name": "view_image", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_image", "output": [
                {"type": "output_text", "text": "Viewed Image"},
                {"type": "input_text", "text": "\nSecond block"},
                {"type": "text", "text": "\nThird block"},
                {"type": "input_file", "file_id": "file_unsupported"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="},
                None,
                {"type": "output_text", "text": None},
            ]},
        ],
        "max_output_tokens": 1,
        "temperature": 0,
    })
    assert res.status_code == 200, res.body
    if endpoint.endswith("/input_tokens"):
        assert res.body["input_tokens"] > 0
    else:
        assert res.body["object"] == "response"
        assert res.body["usage"]["input_tokens"] > 0


# Codex exposes grouped tools (Multi-Agent V1 sub-agent tools, MCP server tools) as "namespace"
# entries. Chat Completions has no grouping, so llama-server flattens a namespaced tool to
# "<namespace>.<tool>", escaping a literal . as "..". The model must see the flattened name and
# a Responses client must get the separate namespace back.
TOOL_CALL_TEMPLATE = str(Path(__file__).resolve().parents[4] / "models" / "templates" /
                                "meta-llama-Llama-3.3-70B-Instruct.jinja")


def function_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
    }


NAMESPACE_TOOLS = [
    {
        "type": "namespace",
        "name": "multi_agent_v1",
        "tools": [
            function_tool("spawn_agent", "Spawn a sub agent"),
            function_tool("wait_agent", "Wait for a sub agent"),
        ],
    },
    {
        "type": "namespace",
        "name": "mcp__files",
        "tools": [function_tool("read.file", "Read a file")],
    },
]

# what the namespace form above must be equivalent to
FLATTENED_TOOLS = [
    function_tool("multi_agent_v1.spawn_agent", "Spawn a sub agent"),
    function_tool("multi_agent_v1.wait_agent", "Wait for a sub agent"),
    function_tool("mcp__files.read..file", "Read a file"),
]


def start_tool_aware_server() -> None:
    global server
    server.jinja = True
    # the tiny test model has no tool-aware template of its own
    server.chat_template_file = TOOL_CALL_TEMPLATE
    server.n_ctx = 4096
    server.start()


def count_prompt_tokens(endpoint: str, data: dict) -> int:
    res = server.make_request("POST", endpoint, data=data)
    assert res.status_code == 200, res.body
    if endpoint.endswith("/input_tokens"):
        return res.body["input_tokens"]
    assert res.body["object"] == "response"
    return res.body["usage"]["input_tokens"]


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/responses/input_tokens"])
def test_responses_namespace_tools_are_flattened(endpoint):
    start_tool_aware_server()
    base = {
        "input": [{"role": "user", "content": "List the files in /tmp"}],
        "max_output_tokens": 1,
        "temperature": 0,
    }

    def count(tools: list | None) -> int:
        data = dict(base)
        if tools is not None:
            data["tools"] = tools
        return count_prompt_tokens(endpoint, data)

    with_namespace = count(NAMESPACE_TOOLS)
    with_flattened = count(FLATTENED_TOOLS)
    without_tools = count(None)

    # a namespace must reach the model as the equivalent list of flattened function tools
    assert with_namespace == with_flattened, f"{with_namespace} != {with_flattened}"
    # ... and those tools must actually be part of the prompt
    assert with_namespace > without_tools, f"{with_namespace} <= {without_tools}"


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/responses/input_tokens"])
def test_responses_namespace_function_call_history(endpoint):
    start_tool_aware_server()
    arguments = '{"input": "work"}'

    def count(function_call: dict) -> int:
        return count_prompt_tokens(endpoint, {
            "input": [
                {"role": "user", "content": "Spawn a worker"},
                function_call,
                {"type": "function_call_output", "call_id": "call_spawn", "output": "worker started"},
            ],
            "tools": NAMESPACE_TOOLS,
            "max_output_tokens": 1,
            "temperature": 0,
        })

    namespaced = count({"type": "function_call", "namespace": "multi_agent_v1", "name": "spawn_agent",
                        "arguments": arguments, "call_id": "call_spawn"})
    flattened = count({"type": "function_call", "name": "multi_agent_v1.spawn_agent",
                       "arguments": arguments, "call_id": "call_spawn"})
    unprefixed = count({"type": "function_call", "name": "spawn_agent",
                        "arguments": arguments, "call_id": "call_spawn"})

    # replayed history must reference the same flattened name the converted tool list advertises
    assert namespaced == flattened, f"{namespaced} != {flattened}"
    # the namespace prefix must not be dropped on replay
    assert namespaced > unprefixed, f"{namespaced} <= {unprefixed}"


def test_responses_namespace_tools_do_not_warn(tmp_path):
    global server
    server.log_path = str(tmp_path / "llama-server.log")
    start_tool_aware_server()

    res = server.make_request("POST", "/v1/responses", data={
        "input": "This is a test",
        "max_output_tokens": 8,
        "temperature": 0,
        "tools": NAMESPACE_TOOLS + [
            {"type": "web_search"},
            {"type": "namespace", "name": "multi_agent_v1", "tools": [
                function_tool("close_agent", "Close a sub agent"),
                {"type": "custom", "name": "not_supported"},
            ]},
        ],
    })
    assert res.status_code == 200, res.body

    events = list(server.make_stream_request("POST", "/v1/responses", data={
        "input": "This is a test",
        "max_output_tokens": 8,
        "temperature": 0,
        "tools": NAMESPACE_TOOLS,
        "stream": True,
    }))
    completed = [e for e in events if e["type"] == "response.completed"]
    assert completed, events
    for item in completed[-1]["response"]["output"]:
        if item.get("type") == "function_call":
            # the flat name is an internal encoding: a Responses client gets namespace + name back
            assert "." not in item["name"], item
            assert "namespace" in item, item

    server.stop()
    log = Path(server.log_path).read_text(encoding="utf-8", errors="replace")

    assert "unsupported Responses tool type 'namespace' skipped" not in log
    # a genuinely unsupported nested tool must still be reported (and proves the log is captured)
    assert "nested in namespace 'multi_agent_v1' skipped" in log

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_responses_stream_with_llama_telemetry():
    global server
    server.n_ctx = 256
    server.n_batch = 32
    server.n_slots = 1
    server.start()

    saw_progress = False
    saw_delta_timings = False
    completed = None

    res = server.make_stream_request("POST", "/responses", data={
        "input": "This is a test" * 10,
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    })

    for data in res:
        if "prompt_progress" in data:
            assert data["type"] == "response.in_progress"
            assert data["prompt_progress"]["total"] > 0
            assert data["prompt_progress"]["processed"] >= data["prompt_progress"]["cache"]
            saw_progress = True
        if "timings" in data:
            assert "prompt_per_second" in data["timings"]
            assert "predicted_per_second" in data["timings"]
            if data["type"] == "response.output_text.delta":
                saw_delta_timings = True
        if data["type"] == "response.completed":
            completed = data

    assert saw_progress
    assert saw_delta_timings
    assert completed is not None
    assert "usage" in completed["response"]
    assert "timings" in completed
