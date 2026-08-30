from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .paths import LabPaths


def schema_path(paths: LabPaths | None = None) -> Path:
    paths = paths or LabPaths.discover()
    return paths.root / "schemas" / "osrlab.schema.json"


def load_schema(paths: LabPaths | None = None) -> dict[str, Any]:
    return json.loads(schema_path(paths).read_text(encoding="utf-8"))


def validate_record(kind: str, record: Mapping[str, Any], paths: LabPaths | None = None) -> None:
    schema = load_schema(paths)
    if kind not in schema["$defs"]:
        raise KeyError(f"Unknown record kind: {kind}")
    validator_schema = {
        "$schema": schema["$schema"],
        "$ref": f"#/$defs/{kind}",
        "$defs": schema["$defs"],
    }
    Draft202012Validator(validator_schema).validate(dict(record))
