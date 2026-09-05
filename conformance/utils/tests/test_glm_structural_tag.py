# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["xgrammar==0.2.3"]
# ///
"""Compile Rust registry-built GLM tags and check their language without a model.

Run from any directory:
    uv run conformance/utils/tests/test_glm_structural_tag.py

The empty tokenizer is for string matching only. No model weights or GPU are
loaded. Both full-string termination and every UTF-8 byte boundary are checked.
"""

import copy
import json
from pathlib import Path
import subprocess
import unittest

try:
    import xgrammar as xgr
except ImportError:
    xgr = None


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "ideas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "ideas"],
    "additionalProperties": False,
}


def tool(strict=True, name="final_answer", schema=None):
    function = {"name": name, "parameters": copy.deepcopy(SCHEMA if schema is None else schema)}
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def call(name="final_answer", ideas="[]", extra="", reasoning="ok"):
    return (
        f"<tool_call>{name}<arg_key>reasoning</arg_key><arg_value>{reasoning}</arg_value>"
        f"<arg_key>ideas</arg_key><arg_value>{ideas}</arg_value>{extra}</tool_call>"
    )


@unittest.skipIf(xgr is None, "run with uv to install pinned xgrammar for the CPU grammar check")
class GlmStructuralTagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        built = subprocess.run(
            ["cargo", "build", "--locked", "-p", "dynamo-parsers", "--example", "check_structural_tag",
             "--message-format=json"],
            cwd=ROOT, text=True, capture_output=True, timeout=300,
        )
        if built.returncode:
            raise RuntimeError(f"Rust conformance adapter did not build:\n{built.stderr}")
        cls.binary = next(
            artifact["executable"] for line in built.stdout.splitlines()
            if (artifact := json.loads(line)).get("executable")
            and artifact.get("target", {}).get("name") == "check_structural_tag"
        )
        cls.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False)

    def build(self, choice="auto", tools=None, reasoning=False, parallel=None, mode="auto", outputs=(), timeout=10):
        request = {
            "parser": "glm47", "tools": [tool()] if tools is None else tools,
            "tool_choice": choice, "starts_in_reasoning": reasoning,
            "parallel_tool_calls": parallel, "schema_mode": mode, "outputs": outputs,
        }
        result = subprocess.run([self.binary], input=json.dumps(request) + "\n",
                                text=True, capture_output=True, check=True, timeout=timeout)
        response = json.loads(result.stdout)
        return self.compiler.compile_structural_tag(response["tag"]), response

    def accepts(self, grammar, text, expected=True):
        # These are COMPLETE candidates; accepting an unfinished prefix is not a pass.
        for chunks in ([text.encode()], [bytes([byte]) for byte in text.encode()]):
            matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
            accepted = all(matcher.accept_string(chunk) for chunk in chunks)
            accepted = accepted and matcher.is_terminated()
            self.assertEqual(expected, accepted, repr(text))

    def test_auto_allows_prose_and_valid_calls_but_rejects_schema_violations(self):
        grammar, _ = self.build()
        for text in ("", "No tool needed.", call(), "I will check. " + call() + " Done."):
            self.accepts(grammar, text)
        for text in (
            call(extra="<arg_key>debug</arg_key><arg_value>true</arg_value>"),
            call(ideas='"[]"'), call(ideas="not-an-array"), call(name="unknown"),
            "<tool_call>final_answer<arg_key>reasoning</arg_key><arg_value>ok</arg_value></tool_call>",
            call()[:-len("</tool_call>")],
        ):
            self.accepts(grammar, text, False)

    def test_required_named_parallel_and_natural_termination(self):
        grammar, _ = self.build(choice="required")
        for text in (call(), call() + " Done.", call() + call()):
            self.accepts(grammar, text)
        for text in ("", "Done without calling a tool."):
            self.accepts(grammar, text, False)
        for choice in ("auto", "required"):
            one, _ = self.build(choice=choice, parallel=False)
            self.accepts(one, call())
            self.accepts(one, call() + call(), False)
        named, _ = self.build(choice={"type": "function", "function": {"name": "selected"}},
                              tools=[tool(), tool(name="selected")])
        self.accepts(named, call(name="selected"))
        self.accepts(named, call(), False)
        self.accepts(named, call(name="selected") * 2, False)

    def test_reasoning_transitions_and_reserved_markers(self):
        for choice in ("auto", "required", {"type": "function", "function": {"name": "final_answer"}}):
            grammar, _ = self.build(choice=choice, reasoning=True)
            self.accepts(grammar, "Let me think 中文. </think>" + call())
            self.accepts(grammar, "</think>" + call())
            self.accepts(grammar, "unfinished reasoning", False)
            self.accepts(grammar, call() + "</think>" + call(), False)
        grammar, _ = self.build()
        for marker in ("</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"):
            self.accepts(grammar, "text " + marker + " stray", False)

    def test_mixed_strict_tools_and_global_schema_mode(self):
        bad = call(extra="<arg_key>debug</arg_key><arg_value>true</arg_value>")
        for strict in (False, None):
            mixed, _ = self.build(tools=[tool(), tool(strict=strict, name="loose")])
            self.accepts(mixed, bad, False)
            self.accepts(mixed, bad.replace("final_answer", "loose"))
            forced, _ = self.build(tools=[tool(strict=strict)], mode="strict")
            self.accepts(forced, bad, False)

    def test_schema_types_round_trip_through_real_glm_parser(self):
        schema = {
            "type": "object", "properties": {
                "s": {"type": "string"}, "n": {"type": "integer"},
                "b": {"type": "boolean"}, "nullable": {"type": ["string", "null"]},
                "obj": {"type": "object", "properties": {"ok": {"enum": ["yes", "no"]}},
                        "required": ["ok"], "additionalProperties": False},
                "arr": {"type": "array", "items": {"type": "integer"}},
            }, "required": ["s", "n", "b", "nullable", "obj", "arr"],
            "additionalProperties": False,
        }
        values = {"s": "hello 中文", "n": 7, "b": True, "nullable": None, "obj": {"ok": "yes"}, "arr": [1, 2]}
        xml = "<tool_call>record" + "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value if isinstance(value, str) else json.dumps(value)}</arg_value>"
            for key, value in values.items()
        ) + "</tool_call>"
        grammar, result = self.build(tools=[tool(name="record", schema=schema)], outputs=[xml])
        self.accepts(grammar, xml)
        calls = result["parsed"][0]["calls"]
        self.assertEqual(1, len(calls))
        self.assertEqual(values, json.loads(calls[0]["function"]["arguments"]))
        self.accepts(grammar, xml.replace('{"ok": "yes"}', '{"ok": "yes", "extra": true}'), False)

    def test_union_coercion_respects_constraints_not_just_types(self):
        for prop, raw, expected in (
            ({"type": ["string", "null"], "enum": ["null"]}, "null", "null"),
            ({"type": ["string", "null"]}, "null", None),
            ({"anyOf": [{"type": "integer", "minimum": 10},
                        {"type": "string", "enum": ["7"]}]}, "7", "7"),
            ({"anyOf": [{"type": "integer", "minimum": 10},
                        {"type": "string", "enum": ["7"]}]}, "12", 12),
            ({"oneOf": [{"type": "boolean", "const": False},
                        {"type": "string", "enum": ["true"]}]}, "true", "true"),
            ({"oneOf": [{"type": "boolean", "const": False},
                        {"type": "string", "enum": ["true"]}]}, "false", False),
            ({"anyOf": [{"type": "array", "minItems": 1, "items": {"type": "integer"}},
                        {"type": "string", "enum": ["[]"]}]}, "[]", "[]"),
            ({"anyOf": [{"type": "object", "properties": {"x": {"type": "integer"}},
                        "required": ["x"], "additionalProperties": False},
                        {"type": "string", "enum": ["{}"]}]}, "{}", "{}"),
        ):
            schema = {"type": "object", "properties": {"value": prop},
                      "required": ["value"], "additionalProperties": False}
            xml = f"<tool_call>record<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
            grammar, result = self.build(tools=[tool(name="record", schema=schema)], outputs=[xml])
            self.accepts(grammar, xml)
            self.assertEqual({"value": expected}, json.loads(result["parsed"][0]["calls"][0]["function"]["arguments"]))

    def test_recursive_reference_coercion_has_bounded_work(self):
        for raw in ("hello", "null", "7"):
            schema = {"type": "object", "properties": {"value": {"$ref": "#/$defs/node"}},
                      "$defs": {"node": {"anyOf": [{"$ref": "#/$defs/node"}] * 3
                                                + [{"type": "string"}]}}}
            xml = f"<tool_call>record<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
            _, result = self.build(tools=[tool(name="record", strict=False, schema=schema)],
                                   outputs=[xml], timeout=2)
            self.assertEqual({"value": raw}, json.loads(result["parsed"][0]["calls"][0]["function"]["arguments"]))
        # Acyclic shared references can also expand exponentially.
        definitions = {"node0": {"type": ["string", "null"]}}
        for index in range(1, 16):
            definitions[f"node{index}"] = {"anyOf": [{"$ref": f"#/$defs/node{index - 1}"}] * 3}
        schema = {"type": "object", "properties": {"value": {"$ref": "#/$defs/node15"}},
                  "$defs": definitions}
        xml = "<tool_call>record<arg_key>value</arg_key><arg_value>null</arg_value></tool_call>"
        _, result = self.build(tools=[tool(name="record", strict=False, schema=schema)],
                               outputs=[xml], timeout=2)
        self.assertEqual({"value": "null"}, json.loads(result["parsed"][0]["calls"][0]["function"]["arguments"]))

    def test_keyword_named_data_and_draft07_ref_siblings(self):
        cases = []
        for key in ("$id", "$ref"):
            cases.append(({"type": "object", "properties": {
                key: {"type": "string"}, "value": {"type": ["integer", "null"]}},
                "required": [key, "value"], "additionalProperties": False},
                {key: "seven", "value": "7"}, {key: "seven", "value": 7}))
        cases.append(({"type": "object", "properties": {
            "value": {"type": ["integer", "null"]}}, "examples": [{"$id": "test"}],
            "required": ["value"], "additionalProperties": False},
            {"value": "7"}, {"value": 7}))
        cases.append(({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
            "definitions": {"int": {"type": "integer"}},
            "properties": {"value": {"$ref": "#/definitions/int", "type": "string"}},
            "required": ["value"], "additionalProperties": False},
            {"value": "7"}, {"value": 7}))
        for schema, values, expected in cases:
            xml = "<tool_call>record" + "".join(
                f"<arg_key>{key}</arg_key><arg_value>{raw}</arg_value>"
                for key, raw in values.items()) + "</tool_call>"
            grammar, result = self.build(tools=[tool(name="record", schema=schema)], outputs=[xml])
            self.accepts(grammar, xml)
            self.assertEqual(expected, json.loads(result["parsed"][0]["calls"][0]["function"]["arguments"]))

    def test_json_looking_strings_and_refs_keep_their_declared_types(self):
        for value in ('[]', '{}', '"quoted"', 'null', '123', 'true', '&quot;', 'x &lt; y'):
            for property_schema in ({"type": "string"}, {"$ref": "#/$defs/text"}):
                schema = {"type": "object", "$defs": {"text": {"type": "string"}},
                          "properties": {"value": property_schema}, "required": ["value"],
                          "additionalProperties": False}
                xml = f"<tool_call>record<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
                grammar, result = self.build(tools=[tool(name="record", schema=schema)], outputs=[xml])
                self.accepts(grammar, xml)
                self.assertEqual({"value": value}, json.loads(result["parsed"][0]["calls"][0]["function"]["arguments"]))


if __name__ == "__main__":
    if xgr is None:
        raise SystemExit("INCONCLUSIVE: xgrammar is missing; run this script with uv run")
    unittest.main(verbosity=2)
