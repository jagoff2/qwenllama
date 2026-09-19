// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include "json.h"

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body, bool allow_image = true);

// Responses exposes grouped tools (Codex Multi-Agent V1 tools, MCP server tools) as a namespace
// tool that contains nested function tools. Chat Completions has no grouping concept, so a namespaced
// tool is presented to the chat layer as one function with a flat, reversible name: every '.' in a
// segment is escaped as .. and the escaped segments are joined by a single unescaped '.' separator.
//   (multi_agent_v1, spawn_agent) -> multi_agent_v1.spawn_agent
//   (foo.bar, baz.qux)             -> foo..bar.baz..qux
// An empty namespace leaves the tool name unchanged. Decoding fails (returns false) for names that
// this encoder could not have produced: no separator, a second separator or an empty segment.
std::string server_chat_encode_namespace_tool_name(const std::string & tool_namespace, const std::string & tool_name);
bool server_chat_decode_namespace_tool_name(const std::string & flat_name, std::string & tool_namespace, std::string & tool_name);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
