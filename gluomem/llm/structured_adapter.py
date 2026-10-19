"""Opt-in, schema-derived normalization; native types remain authoritative."""

from __future__ import annotations

import json
import re
import types
from typing import TYPE_CHECKING, Annotated, Literal, Union, cast, get_args, get_origin

import dspy
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol

    class _AdapterHooks(Protocol):
        """Cooperative next-adapter hooks shared by DSPy's JSON and BAML adapters."""

        def __call__(
            self,
            lm: dspy.BaseLM,
            lm_kwargs: dict[str, Any],
            signature: type[dspy.Signature],
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]: ...

        async def acall(
            self,
            lm: dspy.BaseLM,
            lm_kwargs: dict[str, Any],
            signature: type[dspy.Signature],
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]: ...

        def _prepare_response_format(
            self, lm: dspy.BaseLM, lm_kwargs: dict[str, Any], signature: type[dspy.Signature]
        ) -> None: ...

        def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]: ...


def configured_signature(signature, limits):
    """Clone native output fields with opt-in dot-path string character bounds.

    Only declared model fields are traversed; arrays, unions, and open mappings
    are deliberately unsupported. Empty policy returns the original signature.
    """
    from copy import deepcopy

    from pydantic import Field

    if not isinstance(limits, dict):
        raise ValueError("output_field_limits must be a mapping")
    if not limits:
        return signature

    def constrain(annotation, parts, limit):
        if not parts:
            if annotation is not str:
                raise ValueError("output limit requires a native string field")
            return Annotated[str, Field(max_length=limit)]
        if not isinstance(annotation, type) or not issubclass(annotation, BaseModel):
            raise ValueError("output limit path must traverse declared model fields")
        name, *rest = parts
        if name not in annotation.model_fields:
            raise ValueError("unknown output limit path")
        field = deepcopy(annotation.model_fields[name])
        child = constrain(field.annotation, rest, limit)
        return cast("Callable[..., type[BaseModel]]", create_model)(
            annotation.__name__,
            __base__=annotation,
            __doc__=annotation.__doc__,
            **{name: (child, field)},
        )

    fields = {name: (field.annotation, deepcopy(field)) for name, field in signature.fields.items()}
    for path, limit in sorted(limits.items()):
        if not isinstance(path, str) or not path or type(limit) is not int or limit <= 0:
            raise ValueError(
                "output limits require dot paths and positive integer character counts"
            )
        name, *parts = path.split(".")
        if name not in signature.output_fields:
            raise ValueError("unknown output limit path")
        if not parts:
            raise ValueError("output limits require a string inside a native model field")
        annotation, field = fields[name]
        fields[name] = (constrain(annotation, parts, limit), field)
    return dspy.make_signature(fields, signature.instructions, signature_name=signature.__name__)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _fields(value, fields, *, complete=False):
    if not isinstance(value, dict):
        raise ValueError("object required")
    result = {}
    for key, item in value.items():
        matches = [name for name in fields if name.casefold() == key.casefold()]
        if len(matches) != 1 or matches[0] in result:
            raise ValueError("unknown or ambiguous field")
        name = matches[0]
        result[name] = _normalize(item, fields[name].annotation, complete=complete)
    if any(
        get_origin(field.annotation) is Literal and name not in result
        for name, field in fields.items()
    ):
        raise ValueError("explicit literal required")
    if complete and result.keys() != fields.keys():
        raise ValueError("explicit fields required")
    return result


def _normalize(value, annotation, *, complete=False):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _normalize(value, args[0], complete=complete)
    if origin is Literal and isinstance(value, str):
        matches = [
            x for x in args if isinstance(x, str) and x.casefold() == value.strip().casefold()
        ]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous literal")
        return matches[0]
    if origin in (Union, types.UnionType):
        candidates = []
        for branch in args:
            try:
                candidate = _normalize(value, branch, complete=complete)
                TypeAdapter(branch).validate_python(candidate)
            except ValueError:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("unknown or ambiguous union")
        return candidates[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fields(value, annotation.model_fields, complete=complete)
    if origin is list and isinstance(value, list):
        return [_normalize(x, args[0], complete=complete) for x in value]
    if origin is dict and isinstance(value, dict):
        return {k: _normalize(v, args[1], complete=complete) for k, v in value.items()}
    return value


def output_schema(signature):
    """Native schema plus explicit direct Literal presence, matching _fields."""
    model = cast("Callable[..., type[BaseModel]]", create_model)(
        "GluoMemOutput",
        __config__=ConfigDict(extra="forbid"),
        **{n: (f.annotation, ...) for n, f in signature.output_fields.items()},
    )
    schema = model.model_json_schema()

    def visit(node):
        if not isinstance(node, dict):
            return
        properties = node.get("properties", {})
        required = list(node.get("required", []))
        for name, field in properties.items():
            if ("enum" in field or "const" in field) and name not in required:
                required.append(name)
            visit(field)
        if required:
            node["required"] = required
        for field in node.get("$defs", {}).values():
            visit(field)
        for key in ("items", "additionalProperties"):
            visit(node.get(key))
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for field in node.get(key, []):
                visit(field)

    visit(schema)
    return schema


class _CanonicalParser:
    def __init__(
        self, *, canonicalize=False, schema_mode="auto", schema_strict=False, validation_retries=0
    ):
        super().__init__()
        self.canonicalize = canonicalize
        self.schema_mode = schema_mode
        self.schema_strict = schema_strict
        self.validation_retries = validation_retries

    def _feedback(self, signature, lm_kwargs):
        fields = ",".join(list(signature.output_fields)[:8])[:256]
        instructions = (
            signature.instructions
            + "\nPrevious output: schema_validation at output fields "
            + fields
            + ". Re-evaluate the original inputs against the exact output schema; "
            "do not invent defaults, omit claims, or guess semantic mappings."
        )
        return signature.with_instructions(instructions), {**lm_kwargs, "temperature": 0, "n": 1}

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return cast("_AdapterHooks", super()).__call__(
                lm, dict(lm_kwargs), signature, demos, inputs
            )
        except AdapterParseError as original:
            if not self.validation_retries:
                raise
            repaired, options = self._feedback(signature, lm_kwargs)
            try:
                return cast("_AdapterHooks", super()).__call__(lm, options, repaired, demos, inputs)
            except Exception as final:
                raise final from original

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return await cast("_AdapterHooks", super()).acall(
                lm, dict(lm_kwargs), signature, demos, inputs
            )
        except AdapterParseError as original:
            if not self.validation_retries:
                raise
            repaired, options = self._feedback(signature, lm_kwargs)
            try:
                return await cast("_AdapterHooks", super()).acall(
                    lm, options, repaired, demos, inputs
                )
            except Exception as final:
                raise final from original

    def _prepare_response_format(self, lm, lm_kwargs, signature):
        if self.schema_mode == "auto":
            return cast("_AdapterHooks", super())._prepare_response_format(lm, lm_kwargs, signature)
        if "response_format" not in lm.supported_params or not lm.supports_response_schema:
            raise ValueError("declared response schema capability required")
        # Preserve open mappings. OpenAI's all-objects-closed strict dialect is not assumed.
        lm_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "GluoMemOutput",
                "strict": self.schema_strict,
                "schema": output_schema(signature),
            },
        }

    def parse(self, signature, completion):
        if not self.canonicalize:
            return cast("_AdapterHooks", super()).parse(signature, completion)
        try:
            outputs = signature.output_fields
            text = completion.strip()
            fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
            if fence:
                text = fence[1]
            if len(outputs) == 1:
                tag = re.escape(next(iter(outputs)))
                envelope = re.fullmatch(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
                if envelope:
                    text = envelope[1]
            fields = json.loads(text, object_pairs_hook=_object)
            # Never obtain a bare decision's semantics from model defaults.
            if (
                isinstance(fields, dict)
                and len(outputs) == 1
                and not any(k.casefold() in {n.casefold() for n in outputs} for k in fields)
            ):
                name, field = next(iter(outputs.items()))
                annotation = field.annotation
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    fields = {name: _fields(fields, annotation.model_fields, complete=True)}
            fields = _fields(fields, outputs)
            if fields.keys() != outputs.keys():
                raise ValueError("output fields required")
            return {
                name: TypeAdapter(outputs[name].annotation).validate_python(value)
                for name, value in fields.items()
            }
        except ValueError as exc:
            raise AdapterParseError(
                adapter_name=type(self).__name__,
                signature=signature,
                lm_response="",
                message="structured output validation failed",
            ) from exc


class CanonicalJSONAdapter(_CanonicalParser, dspy.JSONAdapter):
    pass


def make_adapter(
    name, *, canonicalize=False, schema_mode="auto", schema_strict=False, validation_retries=0
):
    options = dict(
        canonicalize=canonicalize,
        schema_mode=schema_mode,
        schema_strict=schema_strict,
        validation_retries=validation_retries,
    )
    enabled = canonicalize or schema_mode != "auto" or validation_retries
    if name == "chat":
        if enabled:
            raise ValueError("structured policy requires json or baml")
        return dspy.ChatAdapter()
    if name == "json":
        return CanonicalJSONAdapter(**options) if enabled else dspy.JSONAdapter()
    if name == "baml":
        from dspy.adapters.baml_adapter import BAMLAdapter

        class CanonicalBAMLAdapter(_CanonicalParser, BAMLAdapter):
            pass

        return CanonicalBAMLAdapter(**options) if enabled else BAMLAdapter()
    raise ValueError("unknown dspy.adapter; expected chat, json or baml")
