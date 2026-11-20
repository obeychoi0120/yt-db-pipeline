"""Video-level train/val/test assignment, stratified by list category.

Existing assignments never change: a new video goes to the split that is furthest below
its target share within the video's category, so adding videos keeps earlier splits fixed.
"""

from __future__ import annotations

import hashlib

from preparation.manifest import SPLITS, category_of


def validate_ratios(ratios) -> dict[str, float]:
    if not isinstance(ratios, dict) or set(ratios) != set(SPLITS):
        raise ValueError(f"split ratios must define exactly {', '.join(SPLITS)}")
    values = {name: float(ratios[name]) for name in SPLITS}
    if any(value < 0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must be non-negative and sum to 1")
    return values


def assign_splits(content_ids, existing, *, seed, ratios) -> dict[str, str]:
    """Return content_id -> split for every id, keeping ``existing`` assignments."""
    ratios = validate_ratios(ratios)
    assigned = {cid: existing[cid] for cid in content_ids if cid in existing}
    counts: dict[str, dict[str, int]] = {}
    for cid, split in assigned.items():
        counts.setdefault(category_of(cid), dict.fromkeys(SPLITS, 0))[split] += 1
    new = sorted(
        (cid for cid in content_ids if cid not in assigned),
        key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest(),
    )
    for cid in new:
        category = counts.setdefault(category_of(cid), dict.fromkeys(SPLITS, 0))
        total = sum(category.values()) + 1
        # Largest deficit against the target count; ties resolve in SPLITS order.
        split = max(SPLITS, key=lambda name: (ratios[name] * total - category[name], -SPLITS.index(name)))
        category[split] += 1
        assigned[cid] = split
    return assigned
