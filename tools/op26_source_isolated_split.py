"""CPU-only, fail-closed, source-connected split for provisional Op26 labels."""
import hashlib
import re
from collections import defaultdict


def ids(row, plural, singular):
    raw = row.get(plural, row.get(singular)) or []
    if not isinstance(raw, list):
        raw = [raw]
    return [str(v).strip() for v in raw if v is not None and str(v).strip()]


def norm(x):
    return " ".join(re.findall(r"\w+", str(x or "").casefold()))


def footprint(row, chunks):
    tenant = str(row.get("tenant_id") or "").strip()
    question = norm(row.get("query_text") or row.get("query"))
    positives = ids(row, "positive_chunk_ids", "positive_chunk_id")
    negatives = ids(row, "negative_chunk_ids", "negative_chunk_id")
    if not tenant or not question or not positives:
        raise ValueError("Missing tenant/query/positive source")
    if len(set(positives + negatives)) != len(positives + negatives):
        raise ValueError("Duplicated or contradictory source IDs")
    keys = {("question", question)}
    for cid in positives + negatives:
        chunk = chunks.get(int(cid)) if cid.isdigit() else chunks.get(cid)
        if chunk is None:
            raise ValueError("Unresolved source ID: " + cid)
        if str(chunk.tenant_id) != tenant:
            raise ValueError("Cross-tenant source ID: " + cid)
        keys.add(("document", tenant, str(chunk.document_id)))
        checksum = str(getattr(chunk.document, "checksum_sha256", "") or "").lower().strip()
        if re.fullmatch("[0-9a-f]{64}", checksum):
            keys.add(("checksum", checksum))
        if norm(chunk.text):
            keys.add(("exact_passage", norm(chunk.text)))
    return keys


def source_isolated_split(rows, chunks, dev_fraction=0.15, seed=42):
    """Never split a component connected by any positive or negative source."""
    if not 0 < dev_fraction < 0.5 or len(rows) < 2:
        raise ValueError("Insufficient split population or invalid dev fraction")
    parent = list(range(len(rows)))

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    owners = {}
    footprints = []
    for index, row in enumerate(rows):
        keys = footprint(row, chunks)
        footprints.append(keys)
        for key in keys:
            prev = owners.setdefault(key, index)
            union(prev, index)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    groups = list(components.values())
    if len(groups) < 2:
        raise ValueError("All source references form one connected component; cannot split")

    def order(group):
        qids = "|".join(sorted(str(rows[i].get("query_id", "")) for i in group))
        digest = hashlib.sha256(f"{seed}:{qids}".encode("utf-8")).hexdigest()
        return len(group), digest

    groups.sort(key=order)
    goal = max(1, round(dev_fraction * len(rows)))
    dev_ids = set()
    for group in groups:
        if len(dev_ids) >= goal:
            break
        if len(dev_ids) + len(group) < len(rows):
            dev_ids.update(group)
    if not dev_ids or len(dev_ids) == len(rows):
        raise ValueError("No nonempty source-isolated train and development partition")
    train_keys = set().union(*(keys for i, keys in enumerate(footprints) if i not in dev_ids))
    dev_keys = set().union(*(keys for i, keys in enumerate(footprints) if i in dev_ids))
    if train_keys & dev_keys:
        raise RuntimeError("Source isolation invariant failed")
    train = [row for i, row in enumerate(rows) if i not in dev_ids]
    dev = [row for i, row in enumerate(rows) if i in dev_ids]
    details = {
        "source_components": len(groups),
        "train_components": sum(all(i not in dev_ids for i in group) for group in groups),
        "dev_components": sum(all(i in dev_ids for i in group) for group in groups),
        "source_footprint_overlap": 0,
        "positive_and_negative_sources_included": True,
        "canonical_families_verified": False,
        "human_label_approval_verified": False,
    }
    return train, dev, details
