#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grouped CSVs WITH DATA from domain_mapping.json + relations.json.

Features:
- Category-based merge: lists of {"table": "..."} in domain_mapping are merged by CompanyRefNo
- Uses relations.json to discover join paths (CompanyMaster ↔ other tables)
- Robust key resolution: handles ProductType_Fk_Id vs ProductTypeId (tokenized matching)
- Drops FK/ID-ish columns automatically, plus SkipFields / optional SkipPatterns
- Keeps identifiers (CompanyRefNo, CompanyNumber)

Outputs:
  - <Group>.csv            (when a group is a list of {"table": ...})
  - <Group>.<Subgroup>.csv (when subgroups exist)

Requires:
  - src.load_config.{load_config, get_processing_params, get_logger}
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import pandas as pd
from src.load_config import get_logger, get_processing_params, load_config

# ====================== Constants ======================
COMPANY_TABLE = "CompanyMaster"
CIN_FIELD = "CompanyRefNo"                 # canonical column in CompanyMaster
COMPANY_NUMBER_SRC_FIELD = "CompanyRefNo"  # exposed as logical "CompanyNumber"
REQ_COL_LOGICAL = ["CompanyRefNo", "CompanyNumber"]

DEFAULT_SKIP_FIELDS: Set[str] = {
    "Final_Submit", "IsActive", "CreatedBy", "CreatedDate", "CreatedIP",
    "UpdatedBy", "UpdatedDate", "UpdatedIP",
}

# These are applied to UNQUALIFIED column names (e.g., "ProductType_Fk_Id", "Id")
DEFAULT_SKIP_PATTERNS = [
    r".*_fk_?id$",   # ProductType_Fk_Id, ProductTypeFK_Id, fkId, etc.
    r"^id$",         # a naked 'Id' column
]

logger = logging.getLogger("grouped_csvs")


# ====================== Utility / Normalization ======================
def norm_table(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    if "." in name:
        return name.split(".", 1)[-1].strip()
    return name


def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def split_tokens(field: str) -> List[str]:
    """Split 'ProductType_Fk_Id' or 'ProductTypeId' -> ['product','type','fk','id'] / ['product','type','id']"""
    s = field.strip()
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    parts = re.split(r'[_\W]+', s)
    return [p.lower() for p in parts if p]


def field_variants(field: str) -> List[str]:
    """Generate normalized variants to match common FK naming (drop 'Fk', add 'Id', etc.)."""
    toks = split_tokens(field)
    variants: List[str] = []
    # base
    variants.append(norm_token("".join(toks)))
    # drop 'fk'
    toks_no_fk = [t for t in toks if t != "fk"]
    variants.append(norm_token("".join(toks_no_fk)))
    # ensure id suffix
    if toks and toks[-1] != "id":
        variants.append(norm_token("".join(toks_no_fk + ["id"])))
    # trunk without last token
    if len(toks_no_fk) > 1:
        variants.append(norm_token("".join(toks_no_fk[:-1])))
    # unique in-order
    out, seen = [], set()
    for v in variants:
        if v and v not in seen:
            out.append(v); seen.add(v)
    return out


def safe_name(seg: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in seg)


def read_json(path: Any) -> Any:
    """Read a JSON file, with logging and simple validation."""
    logger.info(f"Reading JSON: {path}")
    p = Path(path) if isinstance(path, str) else path
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ====================== Relations Graph ======================
class RelationsGraph:
    """
    Stores bidirectional edges between tables with their join keys.
    Each edge: (left_table.left_key) <-> (right_table.right_key)
    """

    def __init__(self):
        # graph[table] -> list of (neighbor_table, this_key, neighbor_key)
        self.graph: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    @staticmethod
    def _normalize_relations(rel_json: Any) -> List[Dict[str, Any]]:
        """
        Support both:
          - Flat list with legacy keys: {"from_table","from_column","to_table","to_column"}
          - Object with "edges": [{"from":{"table","field"},"to":{"table","field"}}]
        """
        edges_raw = rel_json.get("edges") if isinstance(rel_json, dict) and "edges" in rel_json else rel_json
        if not isinstance(edges_raw, list):
            raise ValueError("relations.json must be a list or an object with an 'edges' list.")

        norm: List[Dict[str, Any]] = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            if "from" in e and "to" in e and isinstance(e["from"], dict) and isinstance(e["to"], dict):
                ft = norm_table(e["from"].get("table")); ff = e["from"].get("field")
                tt = norm_table(e["to"].get("table")); tf = e["to"].get("field")
                if ft and ff and tt and tf:
                    norm.append({"from": {"table": ft, "field": ff}, "to": {"table": tt, "field": tf}})
                continue
            if all(k in e for k in ("from_table", "from_column", "to_table", "to_column")):
                ft = norm_table(e["from_table"]); ff = e["from_column"]
                tt = norm_table(e["to_table"]);  tf = e["to_column"]
                if ft and ff and tt and tf:
                    norm.append({"from": {"table": ft, "field": ff}, "to": {"table": tt, "field": tf}})
        return norm

    def load(self, rel_json: Any) -> None:
        edges = self._normalize_relations(rel_json)
        for e in edges:
            ft, ff = e["from"]["table"], e["from"]["field"]
            tt, tf = e["to"]["table"], e["to"]["field"]
            # undirected adjacency (store mapping both ways)
            self.graph[ft].append((tt, ff, tf))
            self.graph[tt].append((ft, tf, ff))
        logger.info(f"Relations: {len(self.graph)} tables, {sum(len(v) for v in self.graph.values())//2} edges")

    def find_path(self, src_table: str, dst_table: str) -> Optional[List[Tuple[str, str, str, str]]]:
        """BFS from src_table to dst_table; returns hops: (left_table,left_key,right_table,right_key)."""
        src_table, dst_table = norm_table(src_table), norm_table(dst_table)
        if src_table == dst_table:
            return []
        visited: Set[str] = {src_table}
        parent: Dict[str, Tuple[str, str, str]] = {}  # child -> (parent_table, parent_key, child_key)
        q: Deque[str] = deque([src_table])
        while q:
            cur = q.popleft()
            for neigh, cur_key, neigh_key in self.graph.get(cur, []):
                if neigh in visited:
                    continue
                visited.add(neigh)
                parent[neigh] = (cur, cur_key, neigh_key)
                if neigh == dst_table:
                    # reconstruct path
                    path: List[Tuple[str, str, str, str]] = []
                    node = dst_table
                    while node != src_table:
                        ptab, pkey, nkey = parent[node]
                        path.append((ptab, pkey, node, nkey))
                        node = ptab
                    path.reverse()
                    return path
                q.append(neigh)
        return None


# ====================== CSV loading ======================
def _pick_csv_for_table(tables_dir: Path, table: str) -> Path:
    table = norm_table(table)
    p1 = tables_dir / f"{table}.csv"
    p2 = tables_dir / f"dbo.{table}.csv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    # fallback: loose match
    tl = table.lower()
    for p in tables_dir.glob("*.csv"):
        if p.stem.lower() == tl or p.name.lower() in (f"{table}.csv".lower(), f"dbo.{table}.csv".lower()):
            return p
    raise FileNotFoundError(f"CSV for table '{table}' not found in {tables_dir}")


def load_table_csv(tables_dir: Path, table: str) -> pd.DataFrame:
    csv_path = _pick_csv_for_table(tables_dir, table)
    # robust encodings
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be"):
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[], encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError(f"Failed to read CSV for {table}: {csv_path}")
    df.columns = [c.strip() for c in df.columns]
    return df


# ====================== Key resolution & joins ======================
def resolve_key(df: pd.DataFrame, table: str, field: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Find/standardize a column in df that corresponds to <table>.<field> and return its name."""
    table = norm_table(table)
    target_fq = f"{table}.{field}"
    target_norm = norm_token(target_fq)

    # exact qualified
    if target_fq in df.columns:
        return df, target_fq
    # ci qualified
    for c in df.columns:
        if c.lower() == target_fq.lower():
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
            return df, target_fq
    # normalized qualified
    for c in df.columns:
        if norm_token(c) == target_norm:
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
            return df, target_fq
    # unqualified exact / ci
    if field in df.columns:
        df = df.rename(columns={field: target_fq})
        return df, target_fq
    for c in df.columns:
        if c.lower() == field.lower():
            df = df.rename(columns={c: target_fq})
            return df, target_fq
    # suffix `<any>.<field>`
    for c in df.columns:
        if c.lower().endswith("." + field.lower()):
            return df, c
    # variants (drop Fk, add Id, etc.)
    for c in df.columns:
        cn = norm_token(c)
        for v in field_variants(field):
            if cn.endswith(v) or cn == v:
                if c != target_fq:
                    df = df.rename(columns={c: target_fq})
                return df, target_fq
    return df, None


def prefix_columns(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Qualify non-required columns as <table>.<col> to avoid collisions."""
    df = df.copy()
    ren = {}
    for c in df.columns:
        if "." in c or c in REQ_COL_LOGICAL:
            continue
        ren[c] = f"{table}.{c}"
    return df.rename(columns=ren)


def ensure_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expose logical identifiers unqualified."""
    if "CompanyRefNo" not in df.columns and f"{COMPANY_TABLE}.{CIN_FIELD}" in df.columns:
        df["CompanyRefNo"] = df[f"{COMPANY_TABLE}.{CIN_FIELD}"]
    if "CompanyNumber" not in df.columns:
        src = f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"
        if src in df.columns:
            df["CompanyNumber"] = df[src]
        elif COMPANY_NUMBER_SRC_FIELD in df.columns:
            df["CompanyNumber"] = df[COMPANY_NUMBER_SRC_FIELD]
    return df


def left_join(base: pd.DataFrame, right: pd.DataFrame, left_key: str, right_key: str) -> pd.DataFrame:
    return base.merge(right, how="left", left_on=left_key, right_on=right_key)


# ====================== Skip rules ======================
def get_unqualified(col: str) -> str:
    return col.split('.', 1)[-1] if '.' in col else col


def compile_skip_patterns(mapping: dict) -> List[re.Pattern]:
    pats = mapping.get("SkipPatterns", []) if isinstance(mapping, dict) else []
    if not pats:
        pats = DEFAULT_SKIP_PATTERNS
    return [re.compile(p, re.IGNORECASE) for p in pats]


def is_fk_or_id_like(unqualified_col: str) -> bool:
    """
    Token-aware FK/ID heuristic:
      - last token is 'id'  -> drop (e.g. ProductId, TypeID)
      - contains 'fk' and last token == 'id'  -> drop (e.g. ProductType_Fk_Id)
    """
    toks = split_tokens(unqualified_col)  # e.g. 'ProductType_Fk_Id' -> ['product','type','fk','id']
    if not toks:
        return False
    if toks[-1] == "id":
        return True
    if "fk" in toks and toks[-1] == "id":
        return True
    return False


def drop_by_skip_rules(
    df: pd.DataFrame,
    skip_fields: Set[str],
    skip_patterns: List[re.Pattern],
    preserve: Set[str] = frozenset({"CompanyRefNo", "CompanyNumber"})
) -> pd.DataFrame:
    """
    Drop columns whose UNQUALIFIED name matches:
      - explicit SkipFields
      - any of SkipPatterns regexes
      - tokenized FK/ID heuristic (see is_fk_or_id_like)
    but always preserve identifiers in `preserve`.
    """
    keep = []
    for c in df.columns:
        uq = get_unqualified(c)
        if uq in preserve:
            keep.append(c)
            continue
        # explicit fields
        if uq in skip_fields:
            continue
        # regex patterns
        if any(rx.search(uq) for rx in skip_patterns):
            continue
        # tokenized FK/ID heuristic
        if is_fk_or_id_like(uq):
            continue
        keep.append(c)
    return df[keep]


def drop_skipfields(df: pd.DataFrame, skip_fields: Set[str]) -> pd.DataFrame:
    """(Legacy) Drop columns whose unqualified name appears in skip_fields (keep keys)."""
    keep = []
    for c in df.columns:
        uq = get_unqualified(c)
        if uq in skip_fields and uq not in {"CompanyRefNo", "CompanyNumber", "Id"}:
            continue
        keep.append(c)
    return df[keep]


def find_companyref_col(df: pd.DataFrame, table: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Try to locate/standardize a 'CompanyRefNo' column in df; return (df, colname-or-None)."""
    candidates = ["CompanyRefNo", f"{table}.CompanyRefNo"]
    # exact / ci
    for cand in candidates:
        if cand in df.columns:
            return df, (cand if cand == "CompanyRefNo" else cand)
        for c in df.columns:
            if c.lower() == cand.lower():
                if c != "CompanyRefNo":
                    df = df.rename(columns={c: "CompanyRefNo"})
                return df, "CompanyRefNo"
    # suffix
    for c in df.columns:
        if c.lower().endswith(".companyrefno") or c.lower() == "companyrefno":
            if c != "CompanyRefNo":
                df = df.rename(columns={c: "CompanyRefNo"})
            return df, "CompanyRefNo"
    # normalized token match
    target = "companyrefno"
    for c in df.columns:
        if norm_token(c) == target:
            if c != "CompanyRefNo":
                df = df.rename(columns={c: "CompanyRefNo"})
            return df, "CompanyRefNo"
    return df, None


# ====================== Builders ======================
def build_dataset_for_subgroup(
    mapping_fields: List[Dict[str, str]],
    graph: RelationsGraph,
    tables_dir: Optional[Path] = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Legacy/field-based builder: given a list of {table, field}, walk relations to join tables,
    and return just requested columns + identifiers.
    """
    # Load base
    base = load_table_csv(tables_dir, COMPANY_TABLE).astype(str)
    base_q = prefix_columns(COMPANY_TABLE, base)
    if f"{COMPANY_TABLE}.{CIN_FIELD}" not in base_q.columns and CIN_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{CIN_FIELD}"] = base_q[CIN_FIELD]
    if f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}" not in base_q.columns and COMPANY_NUMBER_SRC_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"] = base_q[COMPANY_NUMBER_SRC_FIELD]

    working = base_q

    # identify tables needed
    needed_tables: Set[str] = set()
    for item in (mapping_fields or []):
        if isinstance(item, dict) and item.get("table"):
            needed_tables.add(norm_table(item["table"]))
    needed_tables.discard(COMPANY_TABLE)

    # join each table along relation path
    for tbl in needed_tables:
        path = graph.find_path(COMPANY_TABLE, tbl)
        if path is None:
            logger.warning(f"No join path from {COMPANY_TABLE} to {tbl}; skipping.")
            continue
        cur = working
        # determine already joined tables by prefixes
        joined = {c.split('.', 1)[0] for c in cur.columns if '.' in c}
        joined.add(COMPANY_TABLE)

        for lt, lkey, rt, rkey in path:
            if rt in joined:
                continue
            right = load_table_csv(tables_dir, rt).astype(str)
            right_q = prefix_columns(rt, right)
            cur, left_res = resolve_key(cur, lt, lkey)
            right_q, right_res = resolve_key(right_q, rt, rkey)
            if not left_res or not right_res:
                raise KeyError(f"Could not resolve join keys {lt}.{lkey} ~ {rt}.{rkey}")
            cur[left_res] = cur[left_res].astype(str)
            right_q[right_res] = right_q[right_res].astype(str)
            cur = left_join(cur, right_q, left_res, right_res)
            joined.add(rt)
        working = cur

    working = ensure_identifier_columns(working)

    # project requested columns
    out_cols = [c for c in REQ_COL_LOGICAL if c in working.columns]
    req_cols: List[str] = []
    for item in (mapping_fields or []):
        t, f = item.get("table"), item.get("field")
        if not t or not f:
            continue
        fq = f"{norm_table(t)}.{f}"
        if fq in working.columns and fq not in req_cols:
            req_cols.append(fq)
        elif f in working.columns and f not in req_cols:
            req_cols.append(f)
    final_cols = [c for c in out_cols + req_cols if c in working.columns]
    return working[final_cols].copy()


def build_dataset_for_tables_category_merge(
    tables: List[str],
    graph: RelationsGraph,
    tables_dir: Optional[Path],
    skip_fields: Set[str],
    skip_patterns: List[re.Pattern],
    debug: bool = False
) -> pd.DataFrame:
    """
    Category merge: left-join all listed tables onto CompanyMaster by CompanyRefNo.
    If a table lacks CompanyRefNo, walk relations to CompanyMaster, join along the path,
    and then expose CompanyRefNo from CompanyMaster.
    """
    # Base: CompanyMaster
    base = load_table_csv(tables_dir, COMPANY_TABLE).astype(str)
    base_q = prefix_columns(COMPANY_TABLE, base)

    # Ensure qualified identifiers exist
    if f"{COMPANY_TABLE}.{CIN_FIELD}" not in base_q.columns and CIN_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{CIN_FIELD}"] = base_q[CIN_FIELD]
    if f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}" not in base_q.columns and COMPANY_NUMBER_SRC_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"] = base_q[COMPANY_NUMBER_SRC_FIELD]

    # Unqualified logical identifiers
    if "CompanyRefNo" not in base_q.columns and f"{COMPANY_TABLE}.{CIN_FIELD}" in base_q.columns:
        base_q["CompanyRefNo"] = base_q[f"{COMPANY_TABLE}.{CIN_FIELD}"]
    if "CompanyNumber" not in base_q.columns:
        src = f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"
        if src in base_q.columns:
            base_q["CompanyNumber"] = base_q[src]
        elif COMPANY_NUMBER_SRC_FIELD in base_q.columns:
            base_q["CompanyNumber"] = base_q[COMPANY_NUMBER_SRC_FIELD]

    # Start from base
    working = base_q

    for tbl_in in (tables or []):
        tbl = norm_table(tbl_in)
        if not tbl or tbl == COMPANY_TABLE:
            continue

        right_raw = load_table_csv(tables_dir, tbl).astype(str)
        right_q = prefix_columns(tbl, right_raw)

        # If it already has CompanyRefNo, we can join directly
        right_q, right_key = find_companyref_col(right_q, tbl)
        if not right_key:
            # Walk relations to CompanyMaster
            path = graph.find_path(tbl, COMPANY_TABLE)
            if not path:
                logger.warning(f"[CategoryMerge] No path from {tbl} to {COMPANY_TABLE}; skipping.")
                continue

            cur = right_q
            joined = {tbl}
            for lt, lkey, rt, rkey in path:
                if rt in joined:
                    continue
                cur, left_res = resolve_key(cur, lt, lkey)
                rt_df = prefix_columns(rt, load_table_csv(tables_dir, rt).astype(str))
                rt_df, right_resolved = resolve_key(rt_df, rt, rkey)
                if not left_res or not right_resolved:
                    raise KeyError(f"[CategoryMerge] Could not resolve join keys {lt}.{lkey} ~ {rt}.{rkey} while enriching {tbl}")
                cur[left_res] = cur[left_res].astype(str)
                rt_df[right_resolved] = rt_df[right_resolved].astype(str)
                cur = left_join(cur, rt_df, left_res, right_resolved)
                joined.add(rt)

            # Expose CompanyRefNo after reaching CompanyMaster
            if f"{COMPANY_TABLE}.{CIN_FIELD}" in cur.columns and "CompanyRefNo" not in cur.columns:
                cur["CompanyRefNo"] = cur[f"{COMPANY_TABLE}.{CIN_FIELD}"]
            right_q = cur
            right_q, right_key = find_companyref_col(right_q, tbl)

        # Ensure CompanyRefNo column exists for the join
        if "CompanyRefNo" not in right_q.columns and right_key:
            if right_key != "CompanyRefNo" and right_key in right_q.columns:
                right_q["CompanyRefNo"] = right_key if isinstance(right_key, str) else right_q[right_key]
        if "CompanyRefNo" not in right_q.columns:
            logger.warning(f"[CategoryMerge] {tbl}: still no CompanyRefNo; skipping.")
            continue

        # (Optional early) Drop skip-fields on the right side (keep key and Ids)
        right_q = drop_skipfields(right_q, skip_fields)

        # Avoid duplicate CompanyRefNo variants
        qual_key = f"{tbl}.CompanyRefNo"
        if qual_key in right_q.columns and qual_key != "CompanyRefNo":
            right_q = right_q.drop(columns=[qual_key])

        # Merge by CompanyRefNo
        logger.info(f"[CategoryMerge] Joining {tbl} on CompanyRefNo (L={len(working)}, R={len(right_q)})")
        working = working.merge(right_q, how="left", on="CompanyRefNo", suffixes=("", "_dup"))
        dup_cols = [c for c in working.columns if c.endswith("_dup")]
        if dup_cols:
            working = working.drop(columns=dup_cols)

    # Final tidy: drop explicit SkipFields, regex SkipPatterns, and FK/ID-ish columns
    working = drop_by_skip_rules(
        working,
        skip_fields=skip_fields,
        skip_patterns=skip_patterns
    )

    # Reorder: identifiers up front
    id_cols = [c for c in ["CompanyRefNo", "CompanyNumber", f"{COMPANY_TABLE}.{CIN_FIELD}"] if c in working.columns]
    other_cols = [c for c in working.columns if c not in id_cols]
    return working[id_cols + other_cols]


# ====================== Main ======================
def main():
    global logger
    logger = get_logger()
    config, _ = load_config()

    logger.info("=== START: Grouped CSV Generation ===")
    p = get_processing_params(config)

    # Debug logging
    lvl = logging.DEBUG if p.get("debug") else logging.INFO
    logger.setLevel(lvl)
    for h in logger.handlers:
        try:
            h.setLevel(lvl)
        except Exception:
            pass

    dm_path: Path = Path(p["domain_mapping"])
    rel_path: Path = Path(p["table_relations"])
    outdir: Path = Path(p["outdir"])
    tables_dir: Path = Path(p["tables_dir"])

    mapping = read_json(dm_path)
    relations = read_json(rel_path)

    # Build graph from relations
    graph = RelationsGraph()
    graph.load(relations)

    outdir.mkdir(parents=True, exist_ok=True)

    # Skip rules
    skip_fields = set(mapping.get("SkipFields", [])) if isinstance(mapping, dict) else set()
    if not skip_fields:
        skip_fields = set(DEFAULT_SKIP_FIELDS)
    skip_patterns = compile_skip_patterns(mapping)

    manifest: List[Dict[str, Any]] = []

    # Iterate groups
    for group, subs in (mapping or {}).items():
        # Ignore pseudo-groups
        if group in {"SkipFields", "SkipPatterns"}:
            continue

        # Case 1: group is a list of {"table": "..."}  -> single output <Group>.csv
        if isinstance(subs, list) and all(isinstance(it, dict) and "table" in it for it in subs):
            tables = [norm_table(it["table"]) for it in subs]
            logger.info(f"[GROUP] {group}: category merge of tables={tables}")
            df = build_dataset_for_tables_category_merge(
                tables=tables,
                graph=graph,
                tables_dir=tables_dir,
                skip_fields=skip_fields,
                skip_patterns=skip_patterns,
                debug=p.get("debug", False)
            )
            fpath = outdir / f"{safe_name(group)}.csv"
            df.to_csv(fpath, index=False, encoding="utf-8")
            manifest.append({"csv_name": fpath.name, "group": group, "subgroup": None, "rows": len(df), "columns": len(df.columns)})
            continue

        # Case 2: group is a dict of subgroups
        if not isinstance(subs, dict):
            logger.warning(f"Skipping group {group}: value must be dict or list.")
            continue

        logger.info(f"[GROUP] {group}: {len(subs)} subgroups")
        for subgroup, items in subs.items():
            # Subgroup category-merge: list of {"table": "..."} without 'field'
            if isinstance(items, list) and all(isinstance(it, dict) and "table" in it and "field" not in it for it in items):
                tables = [norm_table(it["table"]) for it in items]
                logger.info(f"[SUBGROUP] {group}.{subgroup}: category merge of tables={tables}")
                df = build_dataset_for_tables_category_merge(
                    tables=tables,
                    graph=graph,
                    tables_dir=tables_dir,
                    skip_fields=skip_fields,
                    skip_patterns=skip_patterns,
                    debug=p.get("debug", False)
                )
            else:
                # Legacy field-based mapping: list of {table, field}
                logger.info(f"[SUBGROUP] {group}.{subgroup}: field-based mapping (legacy)")
                df = build_dataset_for_subgroup(
                    mapping_fields=items,
                    graph=graph,
                    tables_dir=tables_dir,
                    debug=p.get("debug", False)
                )

            fpath = outdir / f"{safe_name(group)}.{safe_name(subgroup)}.csv"
            df.to_csv(fpath, index=False, encoding="utf-8")
            manifest.append({"csv_name": fpath.name, "group": group, "subgroup": subgroup, "rows": len(df), "columns": len(df.columns)})

    # Manifest
    if manifest:
        mdf = pd.DataFrame(manifest).sort_values(["group", "subgroup", "csv_name"], na_position="last")
        mpath = outdir / "_index.csv"
        mdf.to_csv(mpath, index=False, encoding="utf-8")

    logger.info("=== DONE: Grouped CSV Generation ===")


if __name__ == "__main__":
    main()
