"""Models under test. Each uses its published query/document formatting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    slug: str
    hf_id: str
    kind: str              # "late" (multi-vector) or "dense" (single vector)
    batch_size: int = 64
    document_length: int | None = None   # override the published document length
    pooled_from: str | None = None       # derived by token pooling (bench.compress), not encoded


MODELS = {
    m.slug: m
    for m in (
        Model("denseon", "lightonai/DenseOn", "dense"),
        Model("lateon", "lightonai/LateOn", "late"),
        Model("lateon512", "lightonai/LateOn", "late", document_length=512),
        Model("mxbai32m", "mixedbread-ai/mxbai-edge-colbert-v0-32m", "late", batch_size=128),
        Model("mlateon", "lightonai/mLateOn", "late", batch_size=32),
        Model("lateon_pool2", "lightonai/LateOn", "late", pooled_from="lateon"),
        Model("lateon_pool4", "lightonai/LateOn", "late", pooled_from="lateon"),
    )
}
