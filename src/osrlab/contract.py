from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .paths import LabPaths


def schema_path(paths: LabPaths | None = None) -> Path:
    paths = paths or LabPaths.discover()
    return paths.root / "schemas" / "osrlab.schema.json"


@lru_cache(maxsize=4)
def load_schema(paths: LabPaths | None = None) -> dict[str, Any]:
    return json.loads(schema_path(paths).read_text(encoding="utf-8"))


def validate_record(kind: str, record: Mapping[str, Any], paths: LabPaths | None = None) -> None:
    validator(kind, paths or LabPaths.discover()).validate(dict(record))


@lru_cache(maxsize=32)
def validator(kind: str, paths: LabPaths) -> Draft202012Validator:
    schema = load_schema(paths)
    if kind not in schema["$defs"]:
        raise KeyError(f"Unknown record kind: {kind}")
    validator_schema = {
        "$schema": schema["$schema"],
        "$ref": f"#/$defs/{kind}",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(validator_schema)
