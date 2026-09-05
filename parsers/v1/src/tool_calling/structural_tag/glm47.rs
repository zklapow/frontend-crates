// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7/GLM-5 structural-tag builder.
//!
//! This mirrors xgrammar's `glm_4_7` built-in format used by vLLM's `glm47`
//! parser. GLM serializes object properties as adjacent
//! `<arg_key>/<arg_value>` pairs inside a `<tool_call>{name}` envelope.
//!
//! Port of the xgrammar `get_glm_4_7_structural_tag` grammar (Apache-2.0),
//! revision b5048784c65f70ca20bc5fb640b06c84583b7a92, with Dynamo's documented
//! schema-mode policy and `parallel_tool_calls=false` handling.

use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use crate::tool_calling::ToolChoice;
use serde::Serialize;
use serde_json::Value;
use std::sync::LazyLock;

const TOOL_CALL_BEGIN: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
pub(crate) const THINK_END: &str = "</think>";

const ARG_CONTROL_TOKENS: &[&str] = &["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

pub(crate) static TOOL_CALL_BAN_TOKENS: LazyLock<Vec<String>> =
    LazyLock::new(|| vec![TOOL_CALL_BEGIN.to_string()]);

#[derive(Serialize)]
struct GlmStructuralTag {
    #[serde(rename = "type")]
    kind: &'static str,
    format: GlmFormat,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
// Keep GLM's extra wire fields private: adding fields to the public format
// structs would break downstream struct literals for unrelated parsers.
enum GlmFormat {
    Tag {
        begin: String,
        content: Box<GlmFormat>,
        end: String,
    },
    TriggeredTags {
        triggers: Vec<String>,
        tags: Vec<GlmFormat>,
        excludes: Vec<String>,
        at_least_one: bool,
        stop_after_first: bool,
    },
    Sequence {
        elements: Vec<GlmFormat>,
    },
    JsonSchema {
        json_schema: Value,
        style: &'static str,
    },
    AnyText {
        excludes: Vec<String>,
    },
}

fn reasoning_excludes() -> Vec<String> {
    ["<think>", THINK_END, TOOL_CALL_BEGIN, TOOL_CALL_END]
        .into_iter()
        .chain(ARG_CONTROL_TOKENS.iter().copied())
        .map(str::to_string)
        .collect()
}

fn text_excludes() -> Vec<String> {
    ["<think>", THINK_END, TOOL_CALL_END]
        .into_iter()
        .chain(ARG_CONTROL_TOKENS.iter().copied())
        .map(str::to_string)
        .collect()
}

pub(crate) fn build_glm47(ctx: &ToolCallFormatBuildContext<'_>) -> anyhow::Result<Option<Value>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }

    let tags: Vec<GlmFormat> = tools
        .into_iter()
        .map(|tool| GlmFormat::Tag {
            begin: format!("{TOOL_CALL_BEGIN}{}", tool.name),
            content: Box::new(GlmFormat::JsonSchema {
                json_schema: resolve_tool_schema(tool, ctx.strict_schema()),
                style: "glm_xml",
            }),
            end: TOOL_CALL_END.to_string(),
        })
        .collect();

    let suffix = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        tags.into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("named tool choice resolved no tool"))?
    } else {
        GlmFormat::TriggeredTags {
            triggers: vec![TOOL_CALL_BEGIN.to_string()],
            tags,
            excludes: text_excludes(),
            at_least_one,
            stop_after_first: ctx.stop_after_first(),
        }
    };

    let format = if ctx.starts_in_reasoning {
        GlmFormat::Sequence {
            elements: vec![
                GlmFormat::Tag {
                    begin: String::new(),
                    content: Box::new(GlmFormat::AnyText {
                        excludes: reasoning_excludes(),
                    }),
                    end: THINK_END.to_string(),
                },
                suffix,
            ],
        }
    } else {
        suffix
    };

    serde_json::to_value(GlmStructuralTag {
        kind: "structural_tag",
        format,
    })
    .map(Some)
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::tool_calling::structural_tag::{StructuralTagBuilder, StructuralTagSchemaMode};
    use crate::tool_calling::{ToolCallConfig, ToolDefinition};

    fn tools(strict: Option<bool>) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "final_answer".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "ideas": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["reasoning", "ideas"],
                    "additionalProperties": false
                })),
                strict,
            },
            ToolDefinition {
                name: "lookup".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                })),
                strict,
            },
        ]
    }

    fn build(
        choice: &ToolChoice,
        tools: &[ToolDefinition],
        starts_in_reasoning: bool,
        parallel_tool_calls: Option<bool>,
    ) -> Value {
        StructuralTagBuilder::Glm47
            .build_tool_call_format(&ToolCallFormatBuildContext {
                tool_choice: choice,
                tools,
                parallel_tool_calls,
                schema_mode: StructuralTagSchemaMode::Auto,
                starts_in_reasoning,
            })
            .unwrap()
            .unwrap()
    }

    #[test]
    fn config_registers_glm_builder() {
        assert!(matches!(
            ToolCallConfig::glm47().structural_tag_builder,
            Some(StructuralTagBuilder::Glm47)
        ));
    }

    #[test]
    fn auto_strict_builds_glm_xml_schema_constraint() {
        let tools = tools(Some(true));
        let tag = build(&ToolChoice::Auto, &tools, false, Some(false));
        let format = &tag["format"];

        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(format["triggers"], json!(["<tool_call>"]));
        assert_eq!(format["at_least_one"], false);
        assert_eq!(format["stop_after_first"], true);
        assert!(
            format["excludes"]
                .as_array()
                .unwrap()
                .contains(&json!("</arg_value>"))
        );
        assert_eq!(format["tags"][0]["begin"], "<tool_call>final_answer");
        assert_eq!(format["tags"][0]["end"], "</tool_call>");
        assert_eq!(format["tags"][0]["content"]["style"], "glm_xml");
        assert_eq!(
            format["tags"][0]["content"]["json_schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn auto_non_strict_keeps_arguments_schema_unconstrained() {
        let tools = tools(Some(false));
        let tag = build(&ToolChoice::Auto, &tools, false, None);
        assert_eq!(tag["format"]["tags"][0]["content"]["json_schema"], true);
    }

    #[test]
    fn required_and_named_preserve_choice_semantics() {
        let tools = tools(Some(true));

        let required = build(&ToolChoice::Required, &tools, false, None);
        assert_eq!(required["format"]["type"], "triggered_tags");
        assert_eq!(required["format"]["at_least_one"], true);
        assert_eq!(required["format"]["tags"].as_array().unwrap().len(), 2);

        let named = build(
            &ToolChoice::Named("lookup".to_string()),
            &tools,
            false,
            None,
        );
        assert_eq!(named["format"]["type"], "tag");
        assert_eq!(named["format"]["begin"], "<tool_call>lookup");
    }

    #[test]
    fn reasoning_prefix_excludes_control_tokens_until_think_end() {
        let tools = tools(Some(true));
        let tag = build(&ToolChoice::Auto, &tools, true, None);
        let prefix = &tag["format"]["elements"][0];

        assert_eq!(tag["format"]["type"], "sequence");
        assert_eq!(prefix["type"], "tag");
        assert_eq!(prefix["end"], "</think>");
        assert!(
            prefix["content"]["excludes"]
                .as_array()
                .unwrap()
                .contains(&json!("<tool_call>"))
        );
    }

    #[test]
    fn none_bans_glm_tool_call_start_token() {
        let ban = StructuralTagBuilder::Glm47
            .build_tool_call_ban()
            .unwrap()
            .unwrap();
        assert_eq!(
            ban["format"]["content"]["exclude_tokens"],
            json!(["<tool_call>"])
        );
    }
}
