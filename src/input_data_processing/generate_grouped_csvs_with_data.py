#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grouped CSVs WITH DATA from updated_domain_mapping.json + relations.json.

Now with smarter key resolution:
- Handles variants like ProductType_Fk_Id ↔ ProductTypeId (drops FK token when needed)
- Tokenizes on underscores/camelCase to compare columns by tokens
- Retains previous robust matching (case-insensitive, normalized, qualified/unqualified)
"""
import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Deque
from collections import deque, defaultdict
from src.load_config import (
    load_config, 
    get_processing_params, 
    get_logger
)
import pandas as pd

# ===== Constants =====
COMPANY_TABLE = "CompanyMaster"
CIN_FIELD = "CompanyRefNo"
COMPANY_NUMBER_SRC_FIELD = "CompanyRefNo"  # exposed as logical "CompanyNumber"

REQ_COL_LOGICAL = ["CompanyRefNo", "CompanyNumber"]

# ===== Small helpers =====
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
    """
    Split a field like 'ProductType_Fk_Id' or 'ProductTypeId' into tokens:
      ['product', 'type', 'fk', 'id']  or  ['product', 'type', 'id']
    """
    s = field.strip()
    # Insert separators before capitals (camelCase -> snake-ish)
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    parts = re.split(r'[_\W]+', s)
    return [p.lower() for p in parts if p]

def field_variants(field: str) -> List[str]:
    """
    Produce normalized variants for a field key to match common naming:
    e.g., 'ProductType_Fk_Id' -> ['producttypefkid','producttypeid','producttypefk','producttype']
    """
    toks = split_tokens(field)
    variants: List[str] = []

    # base
    variants.append(norm_token("".join(toks)))

    # drop fk token if present
    toks_no_fk = [t for t in toks if t != "fk"]
    variants.append(norm_token("".join(toks_no_fk)))

    # just id suffix (if id present)
    if toks and toks[-1] == "id":
        variants.append(norm_token("".join(toks)))  # already present
    else:
        variants.append(norm_token("".join(toks + ["id"])))

    # without last token
    if len(toks) > 1:
        variants.append(norm_token("".join(toks[:-1])))

    # de-duplicate while preserving order
    seen = set()
    out = []
    for v in variants:
        if v and v not in seen:
            out.append(v); seen.add(v)
    return out

def read_json(path: Any, logger) -> Any:
    """
    Read a JSON file, accepting either a Path object or a string path.
    """
    logger.debug(f"Reading JSON from {path}")
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        logger.error(f"JSON file not found at: {path}")
        raise FileNotFoundError(f"JSON file not found at: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully read JSON from {path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {path}: {str(e)}")
        raise ValueError(f"Invalid JSON format in {path}: {str(e)}")

def safe_name(seg: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in seg)

# ===== Relations Graph =====
class RelationsGraph:
    def __init__(self):
        self.graph: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    @staticmethod
    def _normalize_relations(relations_json: Any) -> List[Dict[str, Any]]:
        if isinstance(relations_json, dict) and "edges" in relations_json:
            edges_raw = relations_json["edges"]
        else:
            edges_raw = relations_json
        if not isinstance(edges_raw, list):
            raise ValueError("relations.json must be a list of edges or an object with an 'edges' list.")

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
                ft = norm_table(e.get("from_table")); ff = e.get("from_column")
                tt = norm_table(e.get("to_table")); tf = e.get("to_column")
                if ft and ff and tt and tf:
                    norm.append({"from": {"table": ft, "field": ff}, "to": {"table": tt, "field": tf}})
                continue
        return norm

    def load(self, relations_json: Any) -> None:
        edges = self._normalize_relations(relations_json)
        for e in edges:
            ft, ff = e["from"]["table"], e["from"]["field"]
            tt, tf = e["to"]["table"], e["to"]["field"]
            self.graph[ft].append((tt, ff, tf))
            self.graph[tt].append((ft, tf, ff))

    def find_path(self, src_table: str, dst_table: str) -> Optional[List[Tuple[str, str, str, str]]]:
        src_table = norm_table(src_table); dst_table = norm_table(dst_table)
        if src_table == dst_table:
            return []
        visited: Set[str] = {src_table}
        parent: Dict[str, Tuple[str, str, str]] = {}
        q: Deque[str] = deque([src_table])
        while q:
            cur = q.popleft()
            for neigh, cur_key, neigh_key in self.graph.get(cur, []):
                if neigh in visited:
                    continue
                visited.add(neigh)
                parent[neigh] = (cur, cur_key, neigh_key)
                if neigh == dst_table:
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

# ===== CSV / MSSQL Loaders =====
def _pick_csv_for_table(tables_dir: Path, table: str) -> Path:
    table = norm_table(table)
    wanted_exact = f"{table}.csv"
    wanted_dbo   = f"dbo.{table}.csv"

    p1 = tables_dir / wanted_exact
    if p1.exists(): return p1
    p2 = tables_dir / wanted_dbo
    if p2.exists(): return p2

    all_csvs = list(tables_dir.glob("*.csv"))
    tl = table.lower()

    for p in all_csvs:
        if p.name.lower() in (wanted_exact.lower(), wanted_dbo.lower()):
            return p

    candidates = [p for p in all_csvs if p.stem.lower().endswith("." + tl)]
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:
        for p in candidates:
            parts = p.stem.split(".")
            if len(parts) == 2 and parts[1].lower() == tl:
                return p
        return sorted(candidates, key=lambda x: x.name.lower())[0]

    matches = [p for p in all_csvs if p.stem.lower() == tl]
    if matches: return matches[0]

    raise FileNotFoundError(f"CSV for table '{table}' not found in {tables_dir}.")

def load_table_csv(tables_dir: Path, table: str) -> pd.DataFrame:
    csv_path = _pick_csv_for_table(tables_dir, table)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])
    df.columns = [c.strip() for c in df.columns]
    return df

# ===== Key resolution =====
def resolve_key(df: pd.DataFrame, table: str, field: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Try to find a column in df that corresponds to `table.field`.
    If found, rename to "<table>.<field>" to standardize joins.
    """
    table = norm_table(table)
    target_fq = f"{table}.{field}"
    target_norm = norm_token(target_fq)

    # 0) Fast path: exact
    if target_fq in df.columns:
        return df, target_fq

    # 1) Case-insensitive qualified
    for c in df.columns:
        if c.lower() == target_fq.lower():
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
            return df, target_fq

    # 2) Normalized qualified
    for c in df.columns:
        if norm_token(c) == target_norm:
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
            return df, target_fq

    # 3) Unqualified direct matches
    if field in df.columns:
        df = df.rename(columns={field: target_fq})
        return df, target_fq
    for c in df.columns:
        if c.lower() == field.lower():
            df = df.rename(columns={c: target_fq})
            return df, target_fq

    # 4) Suffix / normalized-suffix
    for c in df.columns:
        if c.lower().endswith("." + field.lower()):
            return df, c

    # 5) Variant-based: allow dropping 'Fk', adding 'Id', etc.
    variants = field_variants(field)
    for c in df.columns:
        cn = norm_token(c)
        for v in variants:
            # Accept if the column endswith the variant or equals it
            if cn.endswith(v) or cn == v:
                # Standardize name to the requested target_fq
                if c != target_fq:
                    df = df.rename(columns={c: target_fq})
                return df, target_fq

    return df, None

def field_variants(field: str) -> List[str]:
    toks = split_tokens(field)
    variants: List[str] = []
    # base
    variants.append(norm_token("".join(toks)))
    # drop fk token
    toks_no_fk = [t for t in toks if t != "fk"]
    variants.append(norm_token("".join(toks_no_fk)))
    # ensure id suffix
    if toks and toks[-1] != "id":
        variants.append(norm_token("".join(toks_no_fk + ["id"])))
    # just head tokens
    if len(toks_no_fk) > 1:
        variants.append(norm_token("".join(toks_no_fk[:-1])))
    # unique
    seen = set(); out = []
    for v in variants:
        if v and v not in seen:
            out.append(v); seen.add(v)
    return out

def split_tokens(field: str) -> List[str]:
    s = field.strip()
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    parts = re.split(r'[_\W]+', s)
    return [p.lower() for p in parts if p]

# ===== Join engine =====
def ensure_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "CompanyRefNo" not in df.columns and f"{COMPANY_TABLE}.{CIN_FIELD}" in df.columns:
        df["CompanyRefNo"] = df[f"{COMPANY_TABLE}.{CIN_FIELD}"]
    if "CompanyNumber" not in df.columns:
        src_col = f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"
        if src_col in df.columns:
            df["CompanyNumber"] = df[src_col]
        elif COMPANY_NUMBER_SRC_FIELD in df.columns:
            df["CompanyNumber"] = df[COMPANY_NUMBER_SRC_FIELD]
    return df

def prefix_columns(table: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {}
    for c in df.columns:
        if "." in c or c in REQ_COL_LOGICAL:
            continue
        new_cols[c] = f"{table}.{c}"
    return df.rename(columns=new_cols)

def left_join(base: pd.DataFrame, right: pd.DataFrame, left_key: str, right_key: str) -> pd.DataFrame:
    return base.merge(right, how="left", left_on=left_key, right_on=right_key)

def build_dataset_for_subgroup(
    mapping_fields: List[Dict[str, str]],
    graph: RelationsGraph,
    tables_dir: Optional[Path] = None,
    logger=None
) -> pd.DataFrame:
    base = load_table_csv(tables_dir, COMPANY_TABLE)
    base = base.astype(str)
    base_q = prefix_columns(COMPANY_TABLE, base)

    if f"{COMPANY_TABLE}.{CIN_FIELD}" not in base_q.columns and CIN_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{CIN_FIELD}"] = base_q[CIN_FIELD]
    if f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}" not in base_q.columns and COMPANY_NUMBER_SRC_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"] = base_q[COMPANY_NUMBER_SRC_FIELD]

    working = base_q

    needed_tables: Set[str] = set()
    for item in (mapping_fields or []):
        if isinstance(item, dict) and item.get("table"):
            needed_tables.add(norm_table(item["table"]))
    needed_tables.discard(COMPANY_TABLE)

    for tbl in needed_tables:
        path = graph.find_path(COMPANY_TABLE, tbl)
        if path is None:
            if logger:
                logger.warning(f"[WARN] No join path from {COMPANY_TABLE} to {tbl}. Skipping.")
            continue

        cur = working
        joined_tables = set([COMPANY_TABLE])  # avoid re-joining the same table (prevents _x/_y suffixes)
        # Detect already joined tables based on qualified column prefixes present in cur
        for col in cur.columns:
            if '.' in col:
                joined_tables.add(col.split('.', 1)[0])

        for (lt, lkey, rt, rkey) in path:
            # If we've already joined `rt`, skip re-merging to avoid column duplication
            if rt in joined_tables:
                if logger:
                    logger.debug(f"[SKIP] {lt}->{rt}: already joined; using existing columns for next hop")
                continue

            right =  load_table_csv(tables_dir, rt)
            right = right.astype(str)
            right_q = prefix_columns(rt, right)

            cur, left_resolved  = resolve_key(cur, lt, lkey)
            right_q, right_resolved = resolve_key(right_q, rt, rkey)

            if not left_resolved or not right_resolved:
                missing = f"left={lt}.{lkey}->{left_resolved}, right={rt}.{rkey}->{right_resolved}"
                raise KeyError(f"Could not resolve join keys: {missing}.\n"
                               f"Left cols (first 20): {list(cur.columns)[:20]}\n"
                               f"Right cols (first 20): {list(right_q.columns)[:20]}")

            cur[left_resolved] = cur[left_resolved].astype(str)
            right_q[right_resolved] = right_q[right_resolved].astype(str)

            if logger:
                logger.debug(f"[JOIN] {lt}.{lkey} == {rt}.{rkey}  →  {left_resolved} == {right_resolved}  (rows L={len(cur)}, R={len(right_q)})")

            cur = left_join(cur, right_q, left_resolved, right_resolved)
            joined_tables.add(rt)
        working = cur

    working = ensure_identifier_columns(working)

    out_cols = [c for c in REQ_COL_LOGICAL if c in working.columns]

    requested_cols: List[str] = []
    for item in (mapping_fields or []):
        table = item.get("table"); field = item.get("field")
        if not table or not field:
            continue
        table = norm_table(table)
        fq = f"{table}.{field}"
        if fq in working.columns and fq not in requested_cols:
            requested_cols.append(fq)
        elif field in working.columns and field not in requested_cols:
            requested_cols.append(field)

    final_cols = [c for c in out_cols + requested_cols if c in working.columns]
    return working[final_cols].copy()

# ===== Main =====
def main():
    # logger & config available whether script or module
    config, logger =  load_config()

    try:
        p = get_processing_params(config)

        # debug level
        lvl = logging.DEBUG if p.get("debug") else logging.INFO
        logger.setLevel(lvl)
        for h in logger.handlers:
            try: h.setLevel(lvl)
            except: pass

        dm, rel, outdir, tables_dir = p["domain_mapping"], p["table_relations"], p["outdir"], p["tables_dir"]
        logger.info("Starting CSV generation…")

        mapping = read_json(dm, logger)
        graph = RelationsGraph()
        graph.load(read_json(rel, logger))
        outdir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict[str, Any]] = []
        for g, subs in (mapping or {}).items():
            if not isinstance(subs, dict): continue
            for s, fields in subs.items():
                try:
                    df = build_dataset_for_subgroup(
                        mapping_fields=fields, graph=graph, tables_dir=tables_dir, logger=logger,
                    )
                    fpath = outdir / f"{safe_name(g)}.{safe_name(s)}.csv"
                    df.to_csv(fpath, index=False, encoding="utf-8")
                    manifest.append({
                        "csv_name": fpath.name, "group": g, "subgroup": s,
                        "rows": len(df), "columns": len(df.columns), "first_cols": list(df.columns[:6]),
                    })
                except Exception as e:
                    logger.error("Build/write failed for %s.%s: %s", g, s, e)

        if manifest:
            pd.DataFrame(manifest).sort_values(["group", "subgroup", "csv_name"]).to_csv(
                outdir / "_index.csv", index=False, encoding="utf-8"
            )
        logger.info("Done. Wrote %d file(s) to %s", len(manifest), outdir)
    except Exception as e:
        logger.exception("main() failed: %s", e)
        raise

if __name__ == "__main__":
    main()
