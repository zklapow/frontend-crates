// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use regex::Regex;
use serde_json::Value;
use std::cell::OnceCell;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::parsed_value::{ParsedValue, coerce_integer_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Render a tool_call block snippet for logs. Bounded so a huge truncated
/// argument body doesn't blow up the log line; control chars are escaped
/// because raw newlines/tabs make the warning unreadable in grep/jq.
fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    let mut out = String::with_capacity(MAX.min(s.len()) + 16);
    let mut bytes = 0usize;
    for ch in s.chars() {
        if bytes >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
        bytes += ch.len_utf8();
    }
    out
}

/// Check if a chunk contains the start of a GLM-4.7 tool call.
/// Format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
pub fn detect_tool_call_start_glm47(chunk: &str, config: &Glm47ParserConfig) -> bool {
    let start_token = &config.tool_call_start;
    let arg_key_start = &config.arg_key_start;

    // Check if we have the complete start token
    if chunk.contains(start_token.as_str()) || chunk.contains(arg_key_start.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming)
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of all consecutive GLM-4.7 tool calls.
/// When a model emits multiple parallel tool calls in one chunk
/// (e.g. `<tool_call>A</tool_call><tool_call>B</tool_call>`), this
/// function advances past every consecutive start→end pair so the
/// entire group is captured as a single jailed region.  Returns the
/// position after the last `</tool_call>` found, or the length of the
/// chunk when no end token is present.
pub fn find_tool_call_end_position_glm47(chunk: &str, config: &Glm47ParserConfig) -> usize {
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !chunk.contains(start_token.as_str()) && chunk.contains(config.arg_key_start.as_str()) {
        return find_bare_glm47_tool_call_end_position(chunk, config).unwrap_or(chunk.len());
    }

    let Some(first_end) = chunk.find(end_token.as_str()) else {
        return chunk.len();
    };

    let mut cursor = first_end + end_token.len();

    loop {
        let rest = &chunk[cursor..];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with(start_token.as_str()) {
            break;
        }
        let trim_offset = rest.len() - trimmed.len();
        let search_from = cursor + trim_offset + start_token.len();
        if let Some(end_pos) = chunk[search_from..].find(end_token.as_str()) {
            cursor = search_from + end_pos + end_token.len();
        } else {
            break;
        }
    }

    cursor
}

fn find_bare_glm47_tool_call_end_position(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    let marker_idx = first_orphan_glm47_marker_index(text, config)?;
    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return None;
    }

    let mut cursor = function_name_start;
    let mut last_complete_end = None;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let call_start = cursor + trim_offset;
        let tail = &text[call_start..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };

        let call_end = call_start + end_pos + config.tool_call_end.len();
        cursor = consume_glm47_close_markers(text, call_end, config);
        last_complete_end = Some(cursor);

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    last_complete_end
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !text.contains(start_token.as_str())
        && let Some(marker_idx) = first_orphan_glm47_marker_index(text, config)
    {
        if let Some((prefix, mut parsed_calls)) =
            recover_bare_glm47_calls(text, marker_idx, config, tools)?
        {
            let recovered_calls = parsed_calls.len();
            warn!(
                why = "bare_body_recovery",
                recovered_calls,
                recovered_bytes = text.len() - prefix.len(),
                kept_prefix_bytes = prefix.len(),
                "GLM-4.7 parser recovered complete bare call body/bodies without <tool_call> start"
            );
            calls.append(&mut parsed_calls);
            return Ok((prefix, calls));
        }
        warn!(
            why = "GLM-4.7 tool-call marker found without <tool_call> start; dropping orphan marker tail so wire tags do not leak into normal_text",
            dropped_block = %truncate_for_log(&text[marker_idx..]),
            "GLM-4.7 parser dropping orphan tool-call marker tail"
        );
        return Ok((orphan_glm47_prefix(text, marker_idx), calls));
    }

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }

            // Only surface normal text that precedes the first parsed call.
            // Text after any </tool_call> is not response content; matches the
            // convention ported into the generic XML parser by PR #9350 and
            // vLLM's glm47_moe_tool_parser.

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block. Unparseable blocks (malformed
                // <tool_call>...</tool_call> markup the parser can't extract)
                // are dropped — emitting the raw markup as normal_text leaks
                // wire tags downstream. vLLM and SGLang both drop on this
                // path; aligning Dynamo to that contract.
                match parse_tool_call_block(block, config, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = %e,
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block = %truncate_for_log(block),
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }

                cursor = abs_end;
            } else {
                // Recovery: outer </tool_call> absent (max_tokens / EOS
                // truncation). Gated on `allow_eof_recovery` so streaming
                // early-exit doesn't fire mid-stream. Also requires an
                // `<arg_key>` opener in the trailing slice as the structural
                // signal that a real tool call was emitted.
                let block = &text[abs_start..];
                let arg_key_start = &config.arg_key_start;
                if config.allow_eof_recovery && block.contains(arg_key_start.as_str()) {
                    match parse_tool_call_block(block, config, tools) {
                        Ok(parsed_call) => {
                            calls.push(parsed_call);
                            cursor = text.len();
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                reason = %e,
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block = %truncate_for_log(block),
                                "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)"
                            );
                        }
                    }
                } else {
                    // Either recovery disabled (production default for GLM-4.7)
                    // or no <arg_key> in the tail (so this is plausibly not a
                    // real tool call at all, just a stray <tool_call> token).
                    let reason = if !config.allow_eof_recovery {
                        "allow_eof_recovery=false (production default for GLM-4.7 to match \
                         vLLM/SGLang on truncated tool calls)"
                    } else {
                        "no <arg_key> in the tail after the <tool_call> start fence, so the \
                         block does not look like a structurally-real GLM-4.7 tool call"
                    };
                    warn!(
                        why = %reason,
                        dropped_block = %truncate_for_log(block),
                        "GLM-4.7 parser dropping truncated tool_call block (no end fence)"
                    );
                }
                // Drop the truncated/unrecoverable tail. Emitting the raw
                // <tool_call>...<arg_key>...<arg_value>... prefix as
                // normal_text would leak wire tags into message.content; vLLM
                // strips the same way on truncation.
                break;
            }
        } else {
            // No more tool calls
            let gap = &text[cursor..];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }
            break;
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

fn recover_bare_glm47_calls_in_span(
    span: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_glm47_marker_index(span, config) else {
        return Ok(None);
    };
    let recovered = recover_bare_glm47_calls(span, marker_idx, config, tools)?;
    if let Some((prefix, parsed_calls)) = recovered {
        let recovered_calls = parsed_calls.len();
        warn!(
            why = "bare_body_gap_recovery",
            recovered_calls,
            recovered_bytes = span.len() - prefix.len(),
            kept_prefix_bytes = prefix.len(),
            "GLM-4.7 parser recovered complete bare call body/bodies before a later <tool_call>"
        );
        return Ok(Some((prefix, parsed_calls)));
    }
    Ok(None)
}

fn first_orphan_glm47_marker_index(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    [
        config.tool_call_end.as_str(),
        config.arg_key_start.as_str(),
        config.arg_key_end.as_str(),
        config.arg_value_start.as_str(),
        config.arg_value_end.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

fn orphan_glm47_prefix(text: &str, marker_idx: usize) -> String {
    let prefix = text[..marker_idx].trim_end();
    let token_start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let tail = &prefix[token_start..];
    if !tail.is_empty()
        && tail
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        prefix[..token_start].trim().to_string()
    } else {
        prefix.trim().to_string()
    }
}

fn recover_bare_glm47_calls(
    text: &str,
    marker_idx: usize,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    if !text[marker_idx..].contains(config.tool_call_end.as_str()) {
        return Ok(None);
    }

    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Ok(None);
    }

    let prefix = text[..function_name_start].to_string();
    let mut cursor = function_name_start;
    let mut calls = Vec::new();

    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        cursor += trim_offset;

        let tail = &text[cursor..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };
        let call_end = cursor + end_pos + config.tool_call_end.len();
        let wrapped = format!("{}{}", config.tool_call_start, &text[cursor..call_end]);
        calls.push(parse_tool_call_block(&wrapped, config, tools)?);
        cursor = call_end;

        if is_glm47_close_marker_spam(&text[cursor..], config) {
            warn!(
                why = "orphan_close_marker_spam",
                dropped_block = %truncate_for_log(&text[cursor..]),
                "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call"
            );
            break;
        }

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }
    Ok(Some((prefix, calls)))
}

fn is_glm47_close_marker_spam(text: &str, config: &Glm47ParserConfig) -> bool {
    let mut rest = text.trim_start();
    let mut saw_close = false;
    while let Some(after_close) = rest.strip_prefix(config.tool_call_end.as_str()) {
        saw_close = true;
        rest = after_close.trim_start();
    }
    saw_close && rest.is_empty()
}

fn consume_glm47_close_markers(text: &str, mut cursor: usize, config: &Glm47ParserConfig) -> usize {
    loop {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let close_start = cursor + trim_offset;
        if !text[close_start..].starts_with(config.tool_call_end.as_str()) {
            return cursor;
        }
        cursor = close_start + config.tool_call_end.len();
    }
}

/// Decode XML character entities in a string.
/// Handles the five predefined XML entities: &lt; &gt; &amp; &quot; &apos;
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Coerce a raw string value using the tool's parameter schema.
/// Falls back to string if no schema is available or the type is unrecognized.
fn coerce_value(raw: &str, schema_type: Option<&str>) -> ParsedValue {
    let trimmed = raw.trim();

    // GLM strings are unquoted XML content, even when they look like JSON.
    // Parsing `[]` or `"hello"` first changes the schema-declared string.
    if schema_type == Some("string") {
        return Value::String(raw.to_string()).into();
    }

    // If the value already looks like JSON (object, array, or quoted string), parse it directly
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(v) = serde_json::from_str::<Value>(trimmed)
    {
        return v.into();
    }

    // Use schema type hints for coercion when available
    match schema_type {
        Some("integer") | Some("int") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
        }
        Some("number") | Some("float") | Some("double") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
            if let Ok(n) = trimmed.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(n)
            {
                return Value::Number(num).into();
            }
        }
        Some("boolean") | Some("bool") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" => return Value::Bool(true).into(),
            "false" | "0" | "no" => return Value::Bool(false).into(),
            _ => {}
        },
        Some("array") => {
            // Try JSON parse first, then fall back to comma-separated splitting
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.is_array()
            {
                return v.into();
            }
            let items: Vec<Value> = trimmed
                .split(',')
                .map(|s| Value::String(s.trim().to_string()))
                .collect();
            return Value::Array(items).into();
        }
        Some("null") if trimmed == "null" || trimmed == "None" || trimmed.is_empty() => {
            return Value::Null.into();
        }
        _ => {}
    }

    Value::String(raw.to_string()).into()
}

/// A schema is request data, never permission to fetch files or network URLs.
/// Keep this explicit even if a downstream crate enables resolver features.
struct NoExternalSchemas;

impl jsonschema::Retrieve for NoExternalSchemas {
    fn retrieve(
        &self,
        _uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schema retrieval is disabled for GLM argument conversion".into())
    }
}

/// Resource IDs change the context in which fragment references are resolved.
fn has_nested_resource_scope(schema: &Value, root: &Value) -> bool {
    let draft4_id = root
        .get("$schema")
        .and_then(Value::as_str)
        .is_some_and(|draft| draft.contains("/draft-04/"))
        && schema.get("id").is_some();
    !std::ptr::eq(schema, root) && (schema.get("$id").is_some() || draft4_id)
}

/// Only a bounded, acyclic local schema is eligible for ambiguous conversion.
/// Count the expanded traversal, not just unique nodes: repeated refs in an
/// acyclic DAG can otherwise cause exponential validation work too.
fn schema_is_bounded_local(root: &Value) -> bool {
    #[derive(Clone, Copy)]
    enum Position {
        Schema,
        SchemaMap,
        SchemaArray,
        Data,
    }
    fn visit<'a>(
        value: &'a Value,
        root: &'a Value,
        path: &mut Vec<&'a Value>,
        remaining: &mut usize,
        position: Position,
    ) -> bool {
        if *remaining == 0
            || path.len() >= 64
            || path.iter().any(|previous| std::ptr::eq(*previous, value))
        {
            return false;
        }
        *remaining -= 1;
        path.push(value);
        let safe = match (value, position) {
            (Value::Object(object), Position::Schema) => {
                // Nested resource scopes / dynamic refs need different reference
                // resolution. Do not send them to a validator after this guard.
                let supported_scope = !has_nested_resource_scope(value, root)
                    && !object.contains_key("$dynamicRef")
                    && !object.contains_key("$recursiveRef");
                let reference_safe = match object.get("$ref") {
                    None => true,
                    Some(Value::String(reference)) => reference
                        .strip_prefix('#')
                        .and_then(|pointer| root.pointer(pointer))
                        .is_some_and(|target| {
                            visit(target, root, path, remaining, Position::Schema)
                        }),
                    _ => false,
                };
                supported_scope
                    && reference_safe
                    && object.iter().all(|(keyword, child)| {
                        let position = match keyword.as_str() {
                            "$defs" | "definitions" | "properties" | "patternProperties"
                            | "dependentSchemas" | "dependencies" => Position::SchemaMap,
                            "allOf" | "anyOf" | "oneOf" | "prefixItems" => Position::SchemaArray,
                            "items" if child.is_array() => Position::SchemaArray,
                            "items"
                            | "additionalItems"
                            | "additionalProperties"
                            | "unevaluatedItems"
                            | "unevaluatedProperties"
                            | "contains"
                            | "propertyNames"
                            | "not"
                            | "if"
                            | "then"
                            | "else"
                            | "contentSchema" => Position::Schema,
                            // Property-name maps and enum/const/examples/default
                            // values are data, even if their keys spell "$ref".
                            _ => Position::Data,
                        };
                        visit(child, root, path, remaining, position)
                    })
            }
            (Value::Object(object), Position::SchemaMap) => object.values().all(|child| {
                // Legacy dependencies can also contain arrays of property names.
                let position = if child.is_array() {
                    Position::Data
                } else {
                    Position::Schema
                };
                visit(child, root, path, remaining, position)
            }),
            (Value::Array(values), Position::SchemaArray) => values
                .iter()
                .all(|child| visit(child, root, path, remaining, Position::Schema)),
            (Value::Object(object), _) => object
                .values()
                .all(|child| visit(child, root, path, remaining, Position::Data)),
            (Value::Array(values), _) => values
                .iter()
                .all(|child| visit(child, root, path, remaining, Position::Data)),
            _ => true,
        };
        path.pop();
        safe
    }
    visit(root, root, &mut Vec::new(), &mut 2048, Position::Schema)
}

/// Concrete types need no candidate disambiguation. Following a single chain
/// keeps this fast path bounded, including cyclic or missing references.
fn concrete_schema_type<'a>(mut schema: &'a Value, root: &'a Value) -> Option<&'a str> {
    for _ in 0..32 {
        if has_nested_resource_scope(schema, root)
            || schema.get("$dynamicRef").is_some()
            || schema.get("$recursiveRef").is_some()
        {
            return None;
        }
        if let Some(reference) = schema.get("$ref") {
            // Draft-07 ignores ref siblings; newer drafts combine them. Let
            // the validator decide whenever siblings could affect the result.
            if schema.as_object()?.len() != 1 {
                return None;
            }
            schema = root.pointer(reference.as_str()?.strip_prefix('#')?)?;
        } else {
            return schema.get("type").and_then(Value::as_str);
        }
    }
    None
}

/// Compile at most once per tool call, and only when a JSON-looking argument
/// has no concrete type. Preserve each property's original root/ref context.
struct ArgumentCoercion<'a> {
    schema: Option<&'a Value>,
    validators: OnceCell<Option<jsonschema::ValidatorMap>>,
}

impl ArgumentCoercion<'_> {
    /// Parent constraints can couple otherwise valid property choices. Try only
    /// the two representations present in GLM's wire format, with a fixed work
    /// budget. Never recursively search an unbounded cross-product of unions.
    fn reconcile_parent_constraints(
        &self,
        arguments: &mut HashMap<String, ParsedValue>,
        raw_values: &HashMap<String, &str>,
    ) {
        let Some(map) = self.validators.get().and_then(Option::as_ref) else {
            return;
        };
        let Some(validator) = map.get("#") else {
            return;
        };
        let Ok(mut instance) = serde_json::to_value(&*arguments) else {
            return;
        };
        if validator.is_valid(&instance) {
            return;
        }
        let Some(root) = self.schema else {
            return;
        };
        let mut alternatives = Vec::new();
        for (key, raw) in raw_values {
            let Some(property) = root.get("properties").and_then(|props| props.get(key)) else {
                continue;
            };
            if concrete_schema_type(property, root).is_some() {
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<Value>(raw.trim()) else {
                continue;
            };
            let pointer = format!(
                "#{}",
                jsonschema::paths::Location::new()
                    .join("properties")
                    .join(key.as_str())
            );
            let Some(property_validator) = map.get(&pointer) else {
                continue;
            };
            let initial = instance[key].clone();
            let other = [parsed, Value::String(raw.to_string())]
                .into_iter()
                .find(|value| value != &initial && property_validator.is_valid(value));
            if let Some(other) = other {
                alternatives.push((key.clone(), [initial, other]));
            }
        }
        alternatives.sort_by(|left, right| left.0.cmp(&right.0));
        fn search(
            validator: &jsonschema::Validator,
            instance: &mut Value,
            alternatives: &[(String, [Value; 2])],
            index: usize,
            remaining: &mut usize,
        ) -> bool {
            if *remaining == 0 {
                return false;
            }
            *remaining -= 1;
            if index == alternatives.len() {
                return validator.is_valid(instance);
            }
            let (key, values) = &alternatives[index];
            for value in values {
                instance[key] = value.clone();
                if search(validator, instance, alternatives, index + 1, remaining) {
                    return true;
                }
            }
            false
        }
        let mut remaining = 256;
        if alternatives.len() <= 32
            && search(validator, &mut instance, &alternatives, 0, &mut remaining)
        {
            for (key, _) in alternatives {
                arguments.insert(key.clone(), instance[&key].clone().into());
            }
        } else if remaining == 0 || alternatives.len() > 32 {
            // No unbounded search when schemas relate many ambiguous fields.
            // Preserve legacy conversions rather than a partially explored choice.
            for (key, _) in alternatives {
                arguments.insert(
                    key.clone(),
                    coerce_value(&decode_xml_entities(raw_values[&key]), None),
                );
            }
        }
    }

    fn coerce(&self, key: &str, raw: &str) -> ParsedValue {
        let Some((root, property)) = self.schema.and_then(|root| {
            root.get("properties")?
                .get(key)
                .map(|property| (root, property))
        }) else {
            return coerce_value(&decode_xml_entities(raw), None);
        };
        if let Some(kind) = concrete_schema_type(property, root) {
            return coerce_value(raw, Some(kind));
        }
        let Ok(json_value) = serde_json::from_str::<Value>(raw.trim()) else {
            return Value::String(raw.to_string()).into();
        };
        let validators = self.validators.get_or_init(|| {
            if !schema_is_bounded_local(root) {
                return None;
            }
            jsonschema::options()
                .with_retriever(NoExternalSchemas)
                .should_validate_formats(false)
                .build_map(root)
                .ok()
        });
        let pointer = format!(
            "#{}",
            jsonschema::paths::Location::new()
                .join("properties")
                .join(key)
        );
        if let Some(validator) = validators.as_ref().and_then(|map| map.get(&pointer)) {
            let literal = Value::String(raw.to_string());
            // GLM strings are unquoted: retain quote characters when the schema
            // accepts them. For non-string JSON prefer the typed value, but only
            // if ALL constraints (enum, const, anyOf, oneOf, bounds, etc.) hold.
            let candidates = if json_value.is_string() {
                [literal, json_value]
            } else {
                [json_value, literal]
            };
            if let Some(value) = candidates
                .into_iter()
                .find(|value| validator.is_valid(value))
            {
                return value.into();
            }
        }
        // This remains a tolerant model-output parser, not a second strict-mode
        // enforcement layer. Unsafe/unsupported schemas and malformed free output
        // retain the historical fallback, without recursive inference or retrieval.
        coerce_value(&decode_xml_entities(raw), None)
    }
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;
    let content = after_start
        .strip_suffix(end_token.as_str())
        .unwrap_or(after_start);

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs
    let mut arguments = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern to match: <arg_key>key</arg_key><arg_value>value</arg_value>
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}{}(.*?){}",
        arg_key_start_escaped, arg_key_end_escaped, arg_value_start_escaped, arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;
    let mut raw_values = HashMap::new();
    let coercion = ArgumentCoercion {
        schema: tools
            .and_then(|tools| tools.iter().find(|tool| tool.name == function_name))
            .and_then(|tool| tool.parameters.as_ref()),
        validators: OnceCell::new(),
    };

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            raw_values.insert(key.to_string(), raw_value);
            arguments.insert(key.to_string(), coercion.coerce(key, raw_value));
        }
    }
    coercion.reconcile_parent_constraints(&mut arguments, &raw_values);

    // Validate function against tools if provided
    if let Some(tools_list) = tools {
        let tool_exists = tools_list.iter().any(|t| t.name == function_name);
        if !tool_exists {
            anyhow::bail!("Function '{}' not found in available tools", function_name);
        }
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&arguments)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    fn parse_with_schema(schema: Value, values: &[(&str, &str)]) -> Value {
        let tools = [ToolDefinition {
            name: "record".to_string(),
            parameters: Some(schema),
            strict: Some(false),
        }];
        let mut raw = "<tool_call>record".to_string();
        for (key, value) in values {
            raw.push_str(&format!(
                "<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
            ));
        }
        raw.push_str("</tool_call>");
        let (calls, _) = try_tool_call_parse_glm47(&raw, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        serde_json::from_str(&calls[0].function.arguments).unwrap()
    }

    #[test]
    fn keyword_named_properties_and_annotations_are_data() {
        for key in ["$id", "$ref", "$dynamicRef", "$recursiveRef"] {
            let mut properties = serde_json::Map::new();
            properties.insert(key.into(), serde_json::json!({"type": "string"}));
            properties.insert(
                "value".into(),
                serde_json::json!({"type": ["integer", "null"]}),
            );
            let schema = serde_json::json!({
                "type": "object", "properties": properties, "examples": [{"$id": "test"}]
            });
            assert!(schema_is_bounded_local(&schema));
            let parsed = parse_with_schema(schema, &[(key, "seven"), ("value", "7")]);
            assert_eq!(parsed["value"], serde_json::json!(7));
            assert_eq!(parsed[key], "seven");
        }
    }

    #[test]
    fn reference_siblings_follow_the_schema_draft() {
        for (draft, expected) in [
            ("http://json-schema.org/draft-07/schema#", Value::Null),
            (
                "https://json-schema.org/draft/2020-12/schema",
                serde_json::json!("null"),
            ),
        ] {
            let schema = serde_json::json!({
                "$schema": draft, "type": "object",
                "definitions": {"nullable": {"type": ["string", "null"]}},
                "properties": {"value": {
                    "$ref": "#/definitions/nullable", "type": "string"
                }}
            });
            assert_eq!(
                parse_with_schema(schema, &[("value", "null")])["value"],
                expected
            );
        }
        let schema = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
            "definitions": {"int": {"type": "integer"}},
            "properties": {"value": {"$ref": "#/definitions/int", "type": "string"}}
        });
        assert_eq!(
            parse_with_schema(schema, &[("value", "7")]),
            serde_json::json!({"value": 7})
        );
    }

    #[test]
    fn parent_constraints_disambiguate_multiple_nullable_arguments() {
        let schema = serde_json::json!({
            "type": "object", "properties": {
                "left": {"type": ["string", "null"]},
                "right": {"type": ["string", "null"]}
            },
            "allOf": [{"properties": {
                "left": {"enum": ["null"]}, "right": {"const": null}
            }}]
        });
        assert_eq!(
            parse_with_schema(schema, &[("left", "null"), ("right", "null")]),
            serde_json::json!({"left": "null", "right": null})
        );
    }

    #[test]
    fn escaped_property_names_and_local_reference_constraints_are_preserved() {
        let schema = serde_json::json!({
            "type": "object", "$defs": {
                "value": {"type": ["string", "null"], "enum": ["null"]}
            },
            "properties": {
                "a/b~c": {"$ref": "#/$defs/value"},
                "snow 雪": {"$ref": "#/$defs/value"}
            }
        });
        assert_eq!(
            parse_with_schema(schema, &[("a/b~c", "null"), ("snow 雪", "null")]),
            serde_json::json!({"a/b~c": "null", "snow 雪": "null"})
        );
    }

    #[test]
    fn cyclic_and_expansive_reference_graphs_do_not_enter_validation() {
        let cyclic = serde_json::json!({
            "$defs": {"node": {"anyOf": [
                {"$ref": "#/$defs/node"}, {"$ref": "#/$defs/node"}, {"$ref": "#/$defs/node"},
                {"type": "string"}
            ]}}, "properties": {"value": {"$ref": "#/$defs/node"}}
        });
        assert!(!schema_is_bounded_local(&cyclic));
        assert_eq!(
            parse_with_schema(cyclic, &[("value", "null")]),
            serde_json::json!({"value": "null"})
        );
        let mut definitions = serde_json::Map::new();
        definitions.insert(
            "node0".into(),
            serde_json::json!({"type": ["string", "null"]}),
        );
        for index in 1..16 {
            let reference = serde_json::json!({"$ref": format!("#/$defs/node{}", index - 1)});
            definitions.insert(
                format!("node{index}"),
                serde_json::json!({"anyOf": [reference.clone(), reference.clone(), reference]}),
            );
        }
        let expansive = serde_json::json!({"$defs": definitions,
            "properties": {"value": {"$ref": "#/$defs/node15"}}});
        assert!(!schema_is_bounded_local(&expansive));
        assert_eq!(
            parse_with_schema(expansive, &[("value", "7")]),
            serde_json::json!({"value": "7"})
        );
    }

    #[test]
    fn candidate_cross_product_is_bounded() {
        let properties: serde_json::Map<String, Value> = (0..33)
            .map(|index| {
                (
                    format!("p{index}"),
                    serde_json::json!({"type": ["string", "null"]}),
                )
            })
            .collect();
        let expected: serde_json::Map<String, Value> = properties
            .keys()
            .map(|key| (key.clone(), serde_json::json!("null")))
            .collect();
        let schema =
            serde_json::json!({"type": "object", "properties": properties, "const": expected});
        let keys: Vec<_> = expected.keys().map(String::as_str).collect();
        let values: Vec<_> = keys.into_iter().map(|key| (key, "null")).collect();
        assert_eq!(parse_with_schema(schema, &values), Value::Object(expected));
    }

    #[test]
    fn schema_conversion_never_retrieves_external_resources() {
        for reference in [
            "https://example.invalid/schema",
            "file:///not-a-schema.json",
        ] {
            let schema = serde_json::json!({"$ref": reference});
            assert!(!schema_is_bounded_local(&schema));
            assert!(
                jsonschema::options()
                    .with_retriever(NoExternalSchemas)
                    .build(&schema)
                    .is_err()
            );
        }
        let nested_scope = serde_json::json!({
            "$defs": {"value": {"type": "integer"}},
            "properties": {"value": {
                "$id": "urn:inner", "$defs": {"value": {"type": "string"}},
                "$ref": "#/$defs/value"
            }}
        });
        assert!(!schema_is_bounded_local(&nested_scope));
        assert_eq!(
            parse_with_schema(nested_scope, &[("value", "7")]),
            serde_json::json!({"value": "7"})
        );
        let draft4_scope = serde_json::json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "definitions": {"value": {"type": "integer"}},
            "properties": {"value": {
                "id": "urn:inner", "definitions": {"value": {"type": "string"}},
                "$ref": "#/definitions/value"
            }}
        });
        assert!(!schema_is_bounded_local(&draft4_scope));
        assert_eq!(
            parse_with_schema(draft4_scope, &[("value", "7")]),
            serde_json::json!({"value": "7"})
        );
    }

    #[test]
    fn json_looking_strings_keep_direct_and_referenced_string_types() {
        for raw in [
            "[]",
            "{}",
            "\"quoted\"",
            "null",
            "123",
            "true",
            "&quot;",
            "x &lt; y",
            "hello 中文",
        ] {
            for property in [
                serde_json::json!({"type": "string"}),
                serde_json::json!({"$ref": "#/$defs/text"}),
            ] {
                let schema = serde_json::json!({
                    "type": "object", "$defs": {"text": {"type": "string"}},
                    "properties": {"value": property}, "required": ["value"],
                    "additionalProperties": false
                });
                assert_eq!(
                    parse_with_schema(schema, &[("value", raw)]),
                    serde_json::json!({"value": raw})
                );
            }
        }
    }

    #[test]
    fn primitive_nullable_and_nested_argument_types_survive_together() {
        let schema = serde_json::json!({
            "type": "object", "properties": {
                "s": {"type": "string"}, "n": {"type": "integer"},
                "b": {"type": "boolean"}, "nullable": {"type": ["string", "null"]},
                "obj": {"type": "object", "properties": {"ok": {"enum": ["yes", "no"]}},
                    "required": ["ok"], "additionalProperties": false},
                "arr": {"type": "array", "items": {"type": "integer"}}
            }, "required": ["s", "n", "b", "nullable", "obj", "arr"],
            "additionalProperties": false
        });
        let values = [
            ("s", "hello 中文"),
            ("n", "7"),
            ("b", "true"),
            ("nullable", "null"),
            ("obj", r#"{"ok":"yes"}"#),
            ("arr", "[1,2]"),
        ];
        assert_eq!(
            parse_with_schema(schema, &values),
            serde_json::json!({
                "s": "hello 中文", "n": 7, "b": true, "nullable": null,
                "obj": {"ok": "yes"}, "arr": [1, 2]
            })
        );
    }

    #[test]
    fn schema_constrained_strings_and_nullable_types_survive_parsing() {
        for (schema, raw, expected) in [
            (
                serde_json::json!({"type": "string"}),
                "[]",
                serde_json::json!("[]"),
            ),
            (
                serde_json::json!({"type": "string"}),
                "\"quoted\"",
                serde_json::json!("\"quoted\""),
            ),
            (
                serde_json::json!({"type": "string"}),
                "&quot;",
                serde_json::json!("&quot;"),
            ),
            (
                serde_json::json!({"type": ["string", "null"]}),
                "null",
                Value::Null,
            ),
            (
                serde_json::json!({"type": ["string", "null"], "enum": ["null"]}),
                "null",
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"anyOf": [
                    {"type": "integer", "minimum": 10},
                    {"type": "string", "enum": ["7"]}
                ]}),
                "7",
                serde_json::json!("7"),
            ),
            (
                serde_json::json!({"oneOf": [
                    {"type": "boolean", "const": false},
                    {"type": "string", "enum": ["true"]}
                ]}),
                "true",
                serde_json::json!("true"),
            ),
            (
                serde_json::json!({"anyOf": [
                    {"type": "integer", "minimum": 10},
                    {"type": "string", "enum": ["7"]}
                ]}),
                "12",
                serde_json::json!(12),
            ),
            (
                serde_json::json!({"oneOf": [
                    {"type": "boolean", "const": false},
                    {"type": "string", "enum": ["true"]}
                ]}),
                "false",
                serde_json::json!(false),
            ),
            (
                serde_json::json!({"anyOf": [
                    {"type": "array", "minItems": 1, "items": {"type": "integer"}},
                    {"type": "string", "enum": ["[]"]}
                ]}),
                "[]",
                serde_json::json!("[]"),
            ),
            (
                serde_json::json!({"anyOf": [
                    {"type": "object", "properties": {"x": {"type": "integer"}},
                        "required": ["x"], "additionalProperties": false},
                    {"type": "string", "enum": ["{}"]}
                ]}),
                "{}",
                serde_json::json!("{}"),
            ),
            (
                serde_json::json!({"anyOf": [{"type": "null"}, {"type": "integer"}]}),
                "7",
                serde_json::json!(7),
            ),
            (
                serde_json::json!({"oneOf": [{"type": "null"}, {"type": "boolean"}]}),
                "true",
                serde_json::json!(true),
            ),
        ] {
            let tools = [ToolDefinition {
                name: "record".to_string(),
                parameters: Some(
                    serde_json::json!({"type": "object", "properties": {"value": schema}}),
                ),
                strict: Some(true),
            }];
            let text = format!(
                "<tool_call>record<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
            );
            let (calls, _) =
                try_tool_call_parse_glm47(&text, &get_test_config(), Some(&tools)).unwrap();
            assert_eq!(calls.len(), 1);
            let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(arguments["value"], expected);
        }
    }

    #[test] // helper
    fn test_detect_tool_call_start() {
        let config = get_test_config();

        // Complete start token
        assert!(detect_tool_call_start_glm47(
            "<tool_call>get_weather",
            &config
        ));

        // Partial start token (streaming)
        assert!(detect_tool_call_start_glm47("Some text <tool", &config));
        assert!(detect_tool_call_start_glm47("Some text <tool_c", &config));

        // No tool call
        assert!(!detect_tool_call_start_glm47("Just normal text", &config));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1
    fn test_parse_simple_tool_call() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(normal_text, Some("".to_string()));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1, TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1, TOOLCALLING.batch.7
    fn test_parse_tool_call_with_multiple_args() {
        let config = get_test_config();
        let message = "<tool_call>book_flight<arg_key>from</arg_key><arg_value>NYC</arg_value><arg_key>to</arg_key><arg_value>LAX</arg_value><arg_key>date</arg_key><arg_value>2026-03-15</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("from").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args.get("to").unwrap().as_str().unwrap(), "LAX");
        assert_eq!(args.get("date").unwrap().as_str().unwrap(), "2026-03-15");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
    fn test_parse_tool_call_with_json_value() {
        let config = get_test_config();
        let message = r#"<tool_call>search<arg_key>filters</arg_key><arg_value>{"category": "books", "price_max": 50}</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let filters = args.get("filters").unwrap();
        assert!(filters.is_object());
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml.
    #[test] // TOOLCALLING.batch.2
    fn test_parse_multiple_tool_calls() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.8
    fn test_parse_with_normal_text() {
        let config = get_test_config();
        let message = "I'll check the weather for you. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal_text,
            Some("I'll check the weather for you. ".to_string())
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.6.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.6.yaml.
    #[test] // TOOLCALLING.batch.6
    fn test_parse_tool_call_no_args() {
        let config = get_test_config();
        let message = "<tool_call>get_current_time</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.is_empty());
    }

    #[test] // helper
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let chunk =
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>more text";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        );
    }

    #[test] // helper — parallel calls: end position must advance past ALL blocks
    fn test_find_tool_call_end_position_parallel() {
        let config = get_test_config();
        let chunk = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>trailing";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            "Must advance past ALL consecutive tool call blocks"
        );
    }

    #[test] // helper — parallel calls with whitespace between blocks
    fn test_find_tool_call_end_position_parallel_with_whitespace() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>\n<tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value></tool_call>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            end_pos,
            chunk.len(),
            "Must handle whitespace/newlines between consecutive blocks"
        );
    }

    #[test] // helper — parallel calls: second block incomplete (streaming)
    fn test_find_tool_call_end_position_parallel_second_incomplete() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call><tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>",
            "Must stop at first complete block when second is incomplete"
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.c, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.2 + TOOLCALLING.batch.8 — bug report repro: text + parallel calls
    fn test_parse_text_then_parallel_calls() {
        let config = get_test_config();
        let message = "I'll check the weather for both cities at the same time!<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>New York</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2, "Both parallel calls must be extracted");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(
            args0.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "New York");

        let text = normal_text.unwrap();
        assert_eq!(
            text,
            "I'll check the weather for both cities at the same time!"
        );
    }

    #[test]
    fn test_parse_two_bare_calls_without_start_markers_recovers_independently() {
        let config = get_test_config();
        let message = "I will check both. get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["timezone"], "EST");
        assert_eq!(normal_text.unwrap(), "I will check both. ");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7, TOOLCALLING.fmt.2
    fn test_parse_multiline_arg_value() {
        let config = get_test_config();
        let message = "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/hello.py</arg_value><arg_key>content</arg_key><arg_value>#!/usr/bin/env python3\nprint(\"Hello, World!\")\n</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("path").unwrap().as_str().unwrap(), "/tmp/hello.py");
        assert!(
            args.contains_key("content"),
            "content argument must be parsed even when it contains newlines"
        );
        let content = args.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("print(\"Hello, World!\")"));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4
    fn test_malformed_tool_call() {
        let config = get_test_config();

        // Missing end tag
        let message = "<tool_call>get_weather";
        let result = try_tool_call_parse_glm47(message, &config, None);
        assert!(result.is_ok()); // Should handle gracefully, no calls extracted

        let (calls, _) = result.unwrap();
        assert_eq!(calls.len(), 0);
    }

    // Recovery for missing outer </tool_call> (max_tokens / EOS truncation):
    // when the inner arg pairs are well-formed, treat EOF as the end token
    // and extract the call. The arg_key opener gates recovery so plain text
    // that happens to start with `<tool_call>` is still preserved verbatim.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_complete_args_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Args complete, only outer </tool_call> missing.
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b, TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_multiple_calls_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Two complete inner calls, missing only the trailing </tool_call> on the second.
        let message = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>EST</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.c, TOOLCALLING.batch.13 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.13.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.batch.8
    fn test_unparseable_block_dropped_no_tag_leak() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        // Tool call block references a function not in the tools list — the
        // whole block (including <tool_call>...<arg_key>...<arg_value>... wire
        // markup) must be dropped, not leaked through normal_text.
        let message = "Here is the result: <tool_call>unknown_func<arg_key>x</arg_key><arg_value>1</arg_value></tool_call> done";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        assert_eq!(calls.len(), 0);
        let text = normal_text.unwrap();
        assert!(
            !text.contains("unknown_func"),
            "Unparseable block must be dropped to avoid tag leakage, got: {text}"
        );
        assert!(
            !text.contains("<tool_call>") && !text.contains("<arg_key>"),
            "Wire-format tags must not leak into normal_text, got: {text}"
        );
        assert!(
            text.contains("Here is the result:") && text.contains("done"),
            "Surrounding prose must be preserved, got: {text}"
        );
    }

    #[test] // helper
    fn test_xml_entity_decoding() {
        let config = get_test_config();
        let message = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>x &lt; y &amp;&amp; y &gt; z</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("content").unwrap().as_str().unwrap(),
            "x < y && y > z"
        );
    }

    #[test] // helper
    fn test_type_coercion_with_schema() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "set_temperature".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "huge_integer": {"type": "integer"},
                    "large_count": {"type": "number"},
                    "huge_count": {"type": "number"},
                    "label": {"type": "string"}
                }
            })),
            strict: None,
        }];

        let message = "<tool_call>set_temperature<arg_key>degrees</arg_key><arg_value>72.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>count</arg_key><arg_value>3</arg_value><arg_key>huge_integer</arg_key><arg_value>9223372036854775808</arg_value><arg_key>large_count</arg_key><arg_value>9007199254740993</arg_value><arg_key>huge_count</arg_key><arg_value>100000000000000000000</arg_value><arg_key>label</arg_key><arg_value>warm</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();

        // number coercion
        assert_eq!(args.get("degrees").unwrap().as_f64().unwrap(), 72.5);
        // boolean coercion
        assert!(args.get("enabled").unwrap().as_bool().unwrap());
        // integer coercion
        assert_eq!(args.get("count").unwrap().as_i64().unwrap(), 3);
        // integer-like numbers should not be rounded through f64
        assert_eq!(
            args.get("large_count").unwrap().as_i64().unwrap(),
            9007199254740993
        );
        let raw_args: HashMap<String, Box<serde_json::value::RawValue>> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(raw_args["huge_integer"].get(), "9223372036854775808");
        assert_eq!(raw_args["huge_count"].get(), "100000000000000000000");
        // string stays string
        assert_eq!(args.get("label").unwrap().as_str().unwrap(), "warm");
    }

    #[test] // helper
    fn test_type_coercion_array_comma_separated() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "tags": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits comma-separated values without JSON brackets
        let message = "<tool_call>tag_item<arg_key>tags</arg_key><arg_value>rust, python, go</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let tags = args.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].as_str().unwrap(), "rust");
        assert_eq!(tags[1].as_str().unwrap(), "python");
        assert_eq!(tags[2].as_str().unwrap(), "go");
    }

    #[test] // helper
    fn test_type_coercion_array_json() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ids": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits proper JSON array
        let message = r#"<tool_call>tag_item<arg_key>ids</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let ids = args.get("ids").unwrap().as_array().unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].as_i64().unwrap(), 1);
    }

    #[test] // helper
    fn test_type_coercion_falls_back_to_string() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "test_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            })),
            strict: None,
        }];

        // "not_a_number" can't be parsed as integer — should fall back to string
        let message = "<tool_call>test_func<arg_key>count</arg_key><arg_value>not_a_number</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            args.get("count").unwrap().is_string(),
            "Should fall back to string when coercion fails"
        );
    }

    /// Parser-level invariant: the glm47 parser is byte-stable — it doesn't
    /// see `finish_reason` and produces the same output regardless of the
    /// upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop / tool_calls
    /// / length mapping) lives in `lib/llm/tests/test_streaming_tool_parsers.rs`
    /// and belongs in the cross-parser finish_reason mapping work-item
    /// (tracked separately).
    #[test]
    fn test_glm47_parser_output_independent_of_upstream_finish() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// TOOLCALLING.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls; normal_text
    /// collapses to the empty string.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.9
    fn test_parse_glm47_empty_and_whitespace_inputs() {
        let config = get_test_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_glm47(input, &config, None).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text (input={:?})",
                input
            );
        }
    }

    /// TOOLCALLING.batch.10 — duplicate calls (same function name twice in one section).
    /// Universal gap noted in the test taxonomy; pin parser-level behavior —
    /// both calls returned with distinct ids.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.10 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.10
    fn test_parse_glm47_duplicate_calls_same_name() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2, "Both duplicate-name calls must be returned");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0.get("location").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "LA");
    }
}
