// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-free adapter for the Python xgrammar conformance test. Each input line
//! supplies tools, a parser name and candidate outputs; each output line contains
//! the registry-built grammar and the real parser's interpretation of candidates.

use std::io::{self, BufRead};

use dynamo_parsers::tool_calling::{
    StructuralTagSchemaMode, ToolCallFormatBuildContext, ToolChoice, ToolDefinition,
    detect_and_parse_tool_call, parsers::get_tool_parser_map,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Function {
    name: String,
    parameters: Option<Value>,
    strict: Option<bool>,
}

#[derive(Deserialize)]
struct Tool {
    function: Function,
}

#[derive(Deserialize)]
struct Case {
    parser: String,
    tools: Vec<Tool>,
    tool_choice: Value,
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
    starts_in_reasoning: bool,
    #[serde(default)]
    outputs: Vec<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let choice = match case.tool_choice.as_str() {
            Some("auto") => ToolChoice::Auto,
            Some("required") => ToolChoice::Required,
            Some("none") => ToolChoice::None,
            _ => ToolChoice::Named(
                case.tool_choice["function"]["name"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("invalid named tool choice"))?
                    .to_string(),
            ),
        };
        let tools: Vec<_> = case
            .tools
            .into_iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            })
            .collect();
        let registry = get_tool_parser_map();
        let builder = registry
            .get(case.parser.as_str())
            .and_then(|config| config.structural_tag_builder.as_ref())
            .ok_or_else(|| anyhow::anyhow!("{} has no structural-tag builder", case.parser))?;
        let tag = builder.build_tool_call_format(&ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: case.parallel_tool_calls,
            schema_mode: case.schema_mode,
            starts_in_reasoning: case.starts_in_reasoning,
        })?;
        let mut parsed = Vec::new();
        for output in &case.outputs {
            let (calls, content) =
                detect_and_parse_tool_call(output, Some(&case.parser), Some(&tools)).await?;
            parsed.push(json!({"calls": calls, "content": content}));
        }
        println!("{}", json!({"tag": tag, "parsed": parsed}));
    }
    Ok(())
}
