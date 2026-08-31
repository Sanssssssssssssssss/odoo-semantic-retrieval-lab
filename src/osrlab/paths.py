from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathBoundaryError(ValueError):
    """Raised when a requested write escapes the laboratory."""


@dataclass(frozen=True)
class LabPaths:
    root: Path

    @classmethod
    def discover(cls) -> "LabPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def docs(self) -> Path:
        return self.root / "corpus" / "raw" / "odoo-documentation-19.0"

    @property
    def snapshot_manifest(self) -> Path:
        return self.root / "corpus" / "raw" / "odoo-documentation-19.0.snapshot.json"

    @property
    def allowed_write_roots(self) -> tuple[Path, ...]:
        return tuple(
            (self.root / name).resolve()
            for name in (
                ".venv",
                ".venv-gpu",
                ".cache",
                ".private",
                ".tools",
                "artifacts",
                "indexes",
                "benchmarks",
                "reviews",
                "corpus/derived",
            )
        )

    def require_write_path(self, candidate: str | Path) -> Path:
        raw = Path(candidate)
        if ".." in raw.parts:
            raise PathBoundaryError(f"Parent traversal is forbidden in write paths: {candidate}")
        resolved = (raw if raw.is_absolute() else self.root / raw).resolve()
        if any(resolved == base or resolved.is_relative_to(base) for base in self.allowed_write_roots):
            if resolved == self.docs.resolve() or resolved.is_relative_to(self.docs.resolve()):
                raise PathBoundaryError(f"Official source checkout is read-only: {resolved}")
            return resolved
        raise PathBoundaryError(f"Write path escapes laboratory allowlist: {resolved}")
