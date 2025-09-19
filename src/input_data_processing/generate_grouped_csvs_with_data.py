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

def read_json(path: Any) -> Any:
    """
    Read a JSON file, accepting either a Path object or a string path.
    """
    logger.info(f"Attempting to read JSON file from: {path}")
    if isinstance(path, str):
        path = Path(path)
        logger.debug(f"Converted string path to Path object: {path}")
    
    if not path.is_file():
        logger.error(f"JSON file not found at: {path}")
        raise FileNotFoundError(f"JSON file not found at: {path}")
    
    logger.debug(f"JSON file exists, proceeding to read: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully read JSON from {path} - loaded {len(data) if isinstance(data, (dict, list)) else 'N/A'} items")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {path}: {str(e)}")
        raise ValueError(f"Invalid JSON format in {path}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error reading JSON from {path}: {str(e)}")
        raise

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
        logger.info("Loading relations graph from JSON data")
        edges = self._normalize_relations(relations_json)
        logger.info(f"Normalized {len(edges)} relation edges")
        
        edge_count = 0
        for e in edges:
            ft, ff = e["from"]["table"], e["from"]["field"]
            tt, tf = e["to"]["table"], e["to"]["field"]
            self.graph[ft].append((tt, ff, tf))
            self.graph[tt].append((ft, tf, ff))
            edge_count += 1
            logger.debug(f"Added bidirectional edge {edge_count}: {ft}.{ff} <-> {tt}.{tf}")
        
        logger.info(f"Successfully loaded relations graph with {len(self.graph)} tables and {edge_count} edges")
        logger.debug(f"Tables in graph: {list(self.graph.keys())}")

    def find_path(self, src_table: str, dst_table: str) -> Optional[List[Tuple[str, str, str, str]]]:
        src_table = norm_table(src_table); dst_table = norm_table(dst_table)
        logger.debug(f"Finding path from {src_table} to {dst_table}")
        
        if src_table == dst_table:
            logger.debug(f"Source and destination are the same table: {src_table}")
            return []
        
        visited: Set[str] = {src_table}
        parent: Dict[str, Tuple[str, str, str]] = {}
        q: Deque[str] = deque([src_table])
        
        while q:
            cur = q.popleft()
            logger.debug(f"Exploring table: {cur}")
            
            for neigh, cur_key, neigh_key in self.graph.get(cur, []):
                if neigh in visited:
                    continue
                visited.add(neigh)
                parent[neigh] = (cur, cur_key, neigh_key)
                logger.debug(f"Found connection: {cur}.{cur_key} -> {neigh}.{neigh_key}")
                
                if neigh == dst_table:
                    logger.debug(f"Destination {dst_table} reached, constructing path")
                    path: List[Tuple[str, str, str, str]] = []
                    node = dst_table
                    while node != src_table:
                        ptab, pkey, nkey = parent[node]
                        path.append((ptab, pkey, node, nkey))
                        node = ptab
                    path.reverse()
                    logger.info(f"Path found from {src_table} to {dst_table}: {len(path)} hops")
                    logger.debug(f"Path details: {path}")
                    return path
                q.append(neigh)
        
        logger.warning(f"No path found from {src_table} to {dst_table}")
        return None

# ===== CSV / MSSQL Loaders =====
def _pick_csv_for_table(tables_dir: Path, table: str) -> Path:
    logger.debug(f"Looking for CSV file for table: {table}")
    table = norm_table(table)
    wanted_exact = f"{table}.csv"
    wanted_dbo   = f"dbo.{table}.csv"
    
    logger.debug(f"Searching for exact matches: {wanted_exact} or {wanted_dbo}")

    p1 = tables_dir / wanted_exact
    if p1.exists(): 
        logger.debug(f"Found exact match: {p1}")
        return p1
    p2 = tables_dir / wanted_dbo
    if p2.exists(): 
        logger.debug(f"Found dbo match: {p2}")
        return p2

    all_csvs = list(tables_dir.glob("*.csv"))
    logger.debug(f"Found {len(all_csvs)} CSV files in directory")
    tl = table.lower()

    for p in all_csvs:
        if p.name.lower() in (wanted_exact.lower(), wanted_dbo.lower()):
            logger.debug(f"Found case-insensitive match: {p}")
            return p

    candidates = [p for p in all_csvs if p.stem.lower().endswith("." + tl)]
    logger.debug(f"Found {len(candidates)} candidates with suffix matching")
    if len(candidates) == 1: 
        logger.debug(f"Single candidate found: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        for p in candidates:
            parts = p.stem.split(".")
            if len(parts) == 2 and parts[1].lower() == tl:
                logger.debug(f"Best candidate found: {p}")
                return p
        sorted_candidate = sorted(candidates, key=lambda x: x.name.lower())[0]
        logger.debug(f"Using first sorted candidate: {sorted_candidate}")
        return sorted_candidate

    matches = [p for p in all_csvs if p.stem.lower() == tl]
    if matches: 
        logger.debug(f"Found stem match: {matches[0]}")
        return matches[0]

    logger.error(f"CSV for table '{table}' not found in {tables_dir}. Available files: {[p.name for p in all_csvs[:10]]}")
    raise FileNotFoundError(f"CSV for table '{table}' not found in {tables_dir}.")

def load_table_csv(tables_dir: Path, table: str) -> pd.DataFrame:
    logger.info(f"Loading CSV data for table: {table}")
    csv_path = _pick_csv_for_table(tables_dir, table)
    logger.debug(f"Selected CSV file: {csv_path}")
    
    # List of encodings to try in order of preference
    encodings_to_try = [
        'utf-8',           # Standard UTF-8
        'utf-8-sig',       # UTF-8 with BOM
        'latin-1',         # ISO-8859-1 (Western European)
        'cp1252',          # Windows-1252 (Western European)
        'iso-8859-1',      # ISO Latin-1
        'cp850',           # DOS Latin-1
        'ascii',           # Basic ASCII
        'utf-16',          # UTF-16 with BOM detection
        'utf-16le',        # UTF-16 Little Endian
        'utf-16be'         # UTF-16 Big Endian
    ]
    
    df = None
    successful_encoding = None
    last_error = None
    
    for encoding in encodings_to_try:
        try:
            logger.debug(f"Attempting to load {csv_path} with encoding: {encoding}")
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[], encoding=encoding)
            successful_encoding = encoding
            logger.info(f"Successfully loaded CSV for {table} with encoding '{encoding}': {len(df)} rows, {len(df.columns)} columns")
            break
        except UnicodeDecodeError as e:
            logger.debug(f"Encoding '{encoding}' failed for {table}: {str(e)}")
            last_error = e
            continue
        except Exception as e:
            logger.debug(f"Unexpected error loading CSV for table {table} with encoding '{encoding}': {str(e)}")
            last_error = e
            continue
    
    if df is None:
        error_msg = f"Failed to load CSV for table {table} from {csv_path} with any of the attempted encodings: {encodings_to_try}"
        logger.error(error_msg)
        if last_error:
            logger.error(f"Last error encountered: {str(last_error)}")
        
        # Try to provide more helpful information
        try:
            file_size = csv_path.stat().st_size
            logger.error(f"File size: {file_size} bytes")
            
            # Read first few bytes to help diagnose encoding
            with open(csv_path, 'rb') as f:
                first_bytes = f.read(100)
                logger.error(f"First 100 bytes (hex): {first_bytes.hex()}")
                logger.error(f"First 100 bytes (repr): {repr(first_bytes)}")
        except Exception as diag_e:
            logger.error(f"Could not read file for diagnostics: {diag_e}")
        
        raise ValueError(error_msg)
    
    # Log encoding information
    if successful_encoding != 'utf-8':
        logger.warning(f"⚠️  CSV file {csv_path} was loaded with encoding '{successful_encoding}' instead of UTF-8.")
        logger.warning(f"⚠️  Consider converting the file to UTF-8 for better compatibility.")
        logger.info(f"💡 To convert to UTF-8, you can use: iconv -f {successful_encoding} -t utf-8 {csv_path} > {csv_path}.utf8")
    
    # Clean column names
    original_columns = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]
    cleaned_columns = df.columns.tolist()
    
    if original_columns != cleaned_columns:
        logger.debug(f"Cleaned column names for {table}")
        logger.debug(f"Original columns: {original_columns[:5]}...")
        logger.debug(f"Cleaned columns: {cleaned_columns[:5]}...")
    
    logger.debug(f"Columns in {table}: {list(df.columns[:10])}")  # Show first 10 columns
    
    # Check for potential encoding issues in the data
    if successful_encoding != 'utf-8':
        # Sample a few cells to check for encoding artifacts
        sample_size = min(100, len(df))
        if sample_size > 0:
            sample_df = df.head(sample_size)
            encoding_issues = 0
            
            for col in sample_df.columns[:5]:  # Check first 5 columns
                for val in sample_df[col].head(10):  # Check first 10 values
                    if isinstance(val, str):
                        # Look for common encoding artifacts
                        if any(char in val for char in ['�', '\ufffd', '\x00']):
                            encoding_issues += 1
            
            if encoding_issues > 0:
                logger.warning(f"⚠️  Detected {encoding_issues} potential encoding artifacts in data. File may need encoding conversion.")
    
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
    
    logger.debug(f"Resolving key: {table}.{field} (target: {target_fq})")
    logger.debug(f"Available columns: {list(df.columns[:15])}")  # Show first 15 columns

    # 0) Fast path: exact
    if target_fq in df.columns:
        logger.debug(f"Found exact match for {target_fq}")
        return df, target_fq

    # 1) Case-insensitive qualified
    for c in df.columns:
        if c.lower() == target_fq.lower():
            logger.debug(f"Found case-insensitive match: {c} -> {target_fq}")
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
                logger.debug(f"Renamed column {c} to {target_fq}")
            return df, target_fq

    # 2) Normalized qualified
    for c in df.columns:
        if norm_token(c) == target_norm:
            logger.debug(f"Found normalized match: {c} -> {target_fq}")
            if c != target_fq:
                df = df.rename(columns={c: target_fq})
                logger.debug(f"Renamed column {c} to {target_fq}")
            return df, target_fq

    # 3) Unqualified direct matches
    if field in df.columns:
        logger.debug(f"Found unqualified direct match: {field} -> {target_fq}")
        df = df.rename(columns={field: target_fq})
        return df, target_fq
    for c in df.columns:
        if c.lower() == field.lower():
            logger.debug(f"Found unqualified case-insensitive match: {c} -> {target_fq}")
            df = df.rename(columns={c: target_fq})
            return df, target_fq

    # 4) Suffix / normalized-suffix
    for c in df.columns:
        if c.lower().endswith("." + field.lower()):
            logger.debug(f"Found suffix match: {c}")
            return df, c

    # 5) Variant-based: allow dropping 'Fk', adding 'Id', etc.
    variants = field_variants(field)
    logger.debug(f"Trying variants for {field}: {variants}")
    for c in df.columns:
        cn = norm_token(c)
        for v in variants:
            # Accept if the column endswith the variant or equals it
            if cn.endswith(v) or cn == v:
                logger.debug(f"Found variant match: {c} (normalized: {cn}) matches variant {v} -> {target_fq}")
                # Standardize name to the requested target_fq
                if c != target_fq:
                    df = df.rename(columns={c: target_fq})
                    logger.debug(f"Renamed column {c} to {target_fq}")
                return df, target_fq

    logger.warning(f"Could not resolve key {table}.{field} in available columns")
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
    debug: bool = False
) -> pd.DataFrame:
    logger.info(f"Building dataset for subgroup with {len(mapping_fields or [])} mapping fields")
    logger.debug(f"Mapping fields: {mapping_fields}")
    
    # Load base company table
    logger.info(f"Loading base table: {COMPANY_TABLE}")
    base = load_table_csv(tables_dir, COMPANY_TABLE)
    base = base.astype(str)
    logger.debug(f"Base table loaded with {len(base)} rows")
    
    base_q = prefix_columns(COMPANY_TABLE, base)
    logger.debug(f"Prefixed columns for {COMPANY_TABLE}")

    # Ensure required identifier columns
    if f"{COMPANY_TABLE}.{CIN_FIELD}" not in base_q.columns and CIN_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{CIN_FIELD}"] = base_q[CIN_FIELD]
        logger.debug(f"Added qualified column: {COMPANY_TABLE}.{CIN_FIELD}")
    if f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}" not in base_q.columns and COMPANY_NUMBER_SRC_FIELD in base_q.columns:
        base_q[f"{COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}"] = base_q[COMPANY_NUMBER_SRC_FIELD]
        logger.debug(f"Added qualified column: {COMPANY_TABLE}.{COMPANY_NUMBER_SRC_FIELD}")

    working = base_q

    # Identify needed tables
    needed_tables: Set[str] = set()
    for item in (mapping_fields or []):
        if isinstance(item, dict) and item.get("table"):
            needed_tables.add(norm_table(item["table"]))
    needed_tables.discard(COMPANY_TABLE)
    
    logger.info(f"Need to join {len(needed_tables)} additional tables: {list(needed_tables)}")

    # Process each needed table
    for tbl in needed_tables:
        logger.info(f"Processing table: {tbl}")
        path = graph.find_path(COMPANY_TABLE, tbl)
        if path is None:
            logger.warning(f"No join path from {COMPANY_TABLE} to {tbl}. Skipping.")
            continue

        logger.info(f"Found join path to {tbl} with {len(path)} hops")
        cur = working
        joined_tables = set([COMPANY_TABLE])  # avoid re-joining the same table (prevents _x/_y suffixes)
        
        # Detect already joined tables based on qualified column prefixes present in cur
        for col in cur.columns:
            if '.' in col:
                joined_tables.add(col.split('.', 1)[0])
        
        logger.debug(f"Already joined tables: {joined_tables}")

        # Execute join path
        for hop_num, (lt, lkey, rt, rkey) in enumerate(path, 1):
            logger.debug(f"Processing hop {hop_num}/{len(path)}: {lt}.{lkey} -> {rt}.{rkey}")
            
            # If we've already joined `rt`, skip re-merging to avoid column duplication
            if rt in joined_tables:
                logger.debug(f"[SKIP] {lt}->{rt}: already joined; using existing columns for next hop")
                continue

            logger.info(f"Joining table {rt} via {lt}.{lkey} = {rt}.{rkey}")
            right = load_table_csv(tables_dir, rt)
            right = right.astype(str)
            right_q = prefix_columns(rt, right)

            cur, left_resolved = resolve_key(cur, lt, lkey)
            right_q, right_resolved = resolve_key(right_q, rt, rkey)

            if not left_resolved or not right_resolved:
                missing = f"left={lt}.{lkey}->{left_resolved}, right={rt}.{rkey}->{right_resolved}"
                logger.error(f"Could not resolve join keys: {missing}")
                logger.error(f"Left cols (first 20): {list(cur.columns)[:20]}")
                logger.error(f"Right cols (first 20): {list(right_q.columns)[:20]}")
                raise KeyError(f"Could not resolve join keys: {missing}.\n"
                               f"Left cols (first 20): {list(cur.columns)[:20]}\n"
                               f"Right cols (first 20): {list(right_q.columns)[:20]}")

            # Ensure string types for join
            cur[left_resolved] = cur[left_resolved].astype(str)
            right_q[right_resolved] = right_q[right_resolved].astype(str)

            logger.info(f"Executing join: {left_resolved} == {right_resolved} (rows L={len(cur)}, R={len(right_q)})")
            cur = left_join(cur, right_q, left_resolved, right_resolved)
            logger.info(f"Join completed: result has {len(cur)} rows, {len(cur.columns)} columns")
            
            joined_tables.add(rt)
        working = cur

    # Ensure identifier columns are present
    logger.debug("Ensuring identifier columns are present")
    working = ensure_identifier_columns(working)

    # Build final column list
    out_cols = [c for c in REQ_COL_LOGICAL if c in working.columns]
    logger.debug(f"Required logical columns found: {out_cols}")

    requested_cols: List[str] = []
    for item in (mapping_fields or []):
        table = item.get("table"); field = item.get("field")
        if not table or not field:
            continue
        table = norm_table(table)
        fq = f"{table}.{field}"
        if fq in working.columns and fq not in requested_cols:
            requested_cols.append(fq)
            logger.debug(f"Added requested column: {fq}")
        elif field in working.columns and field not in requested_cols:
            requested_cols.append(field)
            logger.debug(f"Added requested column (unqualified): {field}")

    final_cols = [c for c in out_cols + requested_cols if c in working.columns]
    logger.info(f"Final dataset: {len(working)} rows, {len(final_cols)} columns")
    logger.debug(f"Final columns: {final_cols}")
    
    return working[final_cols].copy()

# ===== Main =====
def main():
    # logger & config available whether script or module
    global logger, config
    logger = globals().get("logger") or get_logger()
    
    # load_config() returns a tuple (config, logger), so we need to unpack it
    existing_config = globals().get("config")
    if existing_config:
        config = existing_config
    else:
        config, _ = load_config()  # Unpack the tuple to get just the config dict

    logger.info("=== STARTING CSV GENERATION PROCESS ===")
    
    try:
        logger.info("Loading configuration and processing parameters")
        p = get_processing_params(config)

        # debug level
        lvl = logging.DEBUG if p.get("debug") else logging.INFO
        logger.setLevel(lvl)
        logger.info(f"Set logging level to: {logging.getLevelName(lvl)}")
        
        for h in logger.handlers:
            try: 
                h.setLevel(lvl)
                logger.debug(f"Updated handler {type(h).__name__} to level {logging.getLevelName(lvl)}")
            except: 
                pass

        dm, rel, outdir, tables_dir = p["domain_mapping"], p["table_relations"], p["outdir"], p["tables_dir"]
        logger.info(f"Configuration loaded:")
        logger.info(f"  - Domain mapping file: {dm}")
        logger.info(f"  - Relations file: {rel}")
        logger.info(f"  - Output directory: {outdir}")
        logger.info(f"  - Tables directory: {tables_dir}")
        
        logger.info("Starting CSV generation process...")

        # Load domain mapping
        logger.info("Loading domain mapping configuration")
        mapping = read_json(dm)
        logger.info(f"Domain mapping loaded with {len(mapping)} groups")
        
        # Load relations graph
        logger.info("Initializing and loading relations graph")
        graph = RelationsGraph()
        graph.load(read_json(rel))
        
        # Ensure output directory exists
        logger.info(f"Ensuring output directory exists: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {outdir}")

        # Process each group and subgroup
        manifest: List[Dict[str, Any]] = []
        total_groups = len([g for g, subs in (mapping or {}).items() if isinstance(subs, dict)])
        total_subgroups = sum(len(subs) for g, subs in (mapping or {}).items() if isinstance(subs, dict))
        
        logger.info(f"Processing {total_groups} groups with {total_subgroups} total subgroups")
        
        processed_count = 0
        for g, subs in (mapping or {}).items():
            if not isinstance(subs, dict): 
                logger.warning(f"Skipping invalid group {g}: not a dictionary")
                continue
            
            logger.info(f"=== PROCESSING GROUP: {g} ({len(subs)} subgroups) ===")
            
            for s, fields in subs.items():
                processed_count += 1
                logger.info(f"Processing subgroup {processed_count}/{total_subgroups}: {g}.{s}")
                logger.debug(f"Subgroup {g}.{s} has {len(fields) if fields else 0} fields")
                
                # Build dataset - no try/catch, let failures propagate and stop processing
                logger.info(f"Building dataset for {g}.{s}")
                df = build_dataset_for_subgroup(
                    mapping_fields=fields, graph=graph, tables_dir=tables_dir, debug=p.get("debug", False)
                )
                
                # Generate output file path
                fpath = outdir / f"{safe_name(g)}.{safe_name(s)}.csv"
                logger.info(f"*** CREATING NEW FILE: {fpath} ***")
                logger.info(f"File will contain {len(df)} rows and {len(df.columns)} columns")
                logger.debug(f"Columns to be written: {list(df.columns)}")
                
                # Write CSV file
                logger.debug(f"Writing CSV data to {fpath}")
                df.to_csv(fpath, index=False, encoding="utf-8")
                
                # Verify file was created
                if fpath.exists():
                    file_size = fpath.stat().st_size
                    logger.info(f"*** FILE SUCCESSFULLY CREATED: {fpath} ({file_size} bytes) ***")
                else:
                    error_msg = f"*** FILE CREATION FAILED: {fpath} does not exist after write operation ***"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Add to manifest
                manifest_entry = {
                    "csv_name": fpath.name, "group": g, "subgroup": s,
                    "rows": len(df), "columns": len(df.columns), "first_cols": list(df.columns[:6]),
                }
                manifest.append(manifest_entry)
                logger.debug(f"Added to manifest: {manifest_entry}")

        # Create manifest/index file
        if manifest:
            logger.info(f"Creating manifest file with {len(manifest)} entries")
            manifest_df = pd.DataFrame(manifest).sort_values(["group", "subgroup", "csv_name"])
            manifest_path = outdir / "_index.csv"
            
            logger.info(f"*** CREATING MANIFEST FILE: {manifest_path} ***")
            manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
            
            # Verify manifest file was created
            if manifest_path.exists():
                manifest_size = manifest_path.stat().st_size
                logger.info(f"*** MANIFEST FILE SUCCESSFULLY CREATED: {manifest_path} ({manifest_size} bytes) ***")
            else:
                logger.error(f"*** MANIFEST FILE CREATION FAILED: {manifest_path} does not exist ***")
        else:
            logger.warning("No files were successfully generated - skipping manifest creation")

        # Final summary
        logger.info("=== CSV GENERATION PROCESS COMPLETED ===")
        logger.info(f"Successfully processed: {len(manifest)}/{total_subgroups} subgroups")
        logger.info(f"Output directory: {outdir}")
        logger.info(f"Total files created: {len(manifest) + (1 if manifest else 0)}")  # +1 for manifest file
        
        if manifest:
            logger.info("Files created:")
            for entry in manifest:
                logger.info(f"  - {entry['csv_name']} ({entry['rows']} rows, {entry['columns']} cols)")
            logger.info(f"  - _index.csv (manifest file)")
        
    except Exception as e:
        logger.error("=== CSV GENERATION PROCESS FAILED ===")
        logger.exception(f"main() failed with exception: {str(e)}")
        raise

if __name__ == "__main__":
    main()
