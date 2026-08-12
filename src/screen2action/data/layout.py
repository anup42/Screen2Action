"""Environment-configured canonical data-lake paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataLayout:
    """All writable data paths rooted outside tracked source files."""

    root: Path

    @classmethod
    def configured(
        cls,
        root: str | Path | None = None,
        *,
        repository_root: Path | None = None,
    ) -> DataLayout:
        if root is not None:
            selected = Path(root)
        elif os.environ.get("SCREEN2ACTION_DATA_ROOT"):
            selected = Path(os.environ["SCREEN2ACTION_DATA_ROOT"])
        elif repository_root is not None:
            selected = repository_root / "data"
        else:
            selected = Path("data")
        return cls(selected.expanduser().resolve())

    def raw_revision(self, source: str, revision: str) -> Path:
        return self.root / "raw" / source / revision

    def interim_revision(self, source: str, revision: str) -> Path:
        return self.root / "interim" / source / revision

    def normalized(self, dataset_version: str) -> Path:
        return self.root / "normalized" / dataset_version

    def images(self, dataset_version: str) -> Path:
        return self.normalized(dataset_version) / "images"

    def image(self, dataset_version: str, sha256: str, image_format: str = "png") -> Path:
        return self.images(dataset_version) / sha256[:2] / f"{sha256}.{image_format}"

    def table(self, dataset_version: str, name: str) -> Path:
        return self.normalized(dataset_version) / name / f"part-00000-{name}.parquet"

    def audits(self, dataset_version: str) -> Path:
        return self.normalized(dataset_version) / "audits"

    def manifests(self, dataset_version: str) -> Path:
        return self.normalized(dataset_version) / "manifests"

    def shards(self, dataset_version: str) -> Path:
        return self.root / "shards" / dataset_version

    def perception_cache(self, bundle_digest: str) -> Path:
        return self.root / "caches" / "perception" / bundle_digest

    def license_acknowledgements(self) -> Path:
        return self.root / "acknowledgements"
