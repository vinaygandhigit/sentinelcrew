"""A small, dependency-free JSON-schema subset validator.

Used as a guardrail on every structured LLM response. Supported keywords:
``type`` (object, array, string, number, integer, boolean, null or a list of
types), ``properties``, ``required``, ``items``, ``enum``, ``minimum``,
``maximum``, ``minItems``, ``maxItems``, ``maxLength`` and
``additionalProperties: false``.
"""

from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    """Raised when data does not satisfy a schema."""


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(data: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``data`` against ``schema``; raise :class:`SchemaError`."""
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        unknown = [t for t in types if t not in _TYPE_CHECKS]
        if unknown:
            raise SchemaError(f"{path}: unsupported schema type(s) {unknown}")
        if not any(_TYPE_CHECKS[t](data) for t in types):
            raise SchemaError(f"{path}: expected {expected}, got {type(data).__name__}")

    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(f"{path}: value {data!r} not in {schema['enum']}")

    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            raise SchemaError(f"{path}: {data} < minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            raise SchemaError(f"{path}: {data} > maximum {schema['maximum']}")

    if isinstance(data, str) and "maxLength" in schema and len(data) > schema["maxLength"]:
        raise SchemaError(f"{path}: string longer than {schema['maxLength']}")

    if isinstance(data, dict):
        props: dict[str, Any] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                raise SchemaError(f"{path}: missing required property '{key}'")
        if schema.get("additionalProperties") is False:
            extra = set(data) - set(props)
            if extra:
                raise SchemaError(f"{path}: unexpected properties {sorted(extra)}")
        for key, sub_schema in props.items():
            if key in data:
                validate(data[key], sub_schema, f"{path}.{key}")

    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            raise SchemaError(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            raise SchemaError(f"{path}: more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(data):
                validate(item, item_schema, f"{path}[{index}]")
