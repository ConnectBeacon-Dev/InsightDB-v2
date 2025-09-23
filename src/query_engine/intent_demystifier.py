#!/usr/bin/env python3
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------------- I/O ----------------

def load_json(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))

def get_node(obj: dict, path: List[str]) -> Any:
    cur: Any = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def collect_pairs(node: Any) -> List[Tuple[str, str]]:
    """Collect [{'table':..., 'field':...}] recursively → [(table, field), ...]"""
    out: List[Tuple[str, str]] = []
    if isinstance(node, dict):
        if "table" in node and "field" in node:
            t, f = node.get("table"), node.get("field")
            if t and f:
                out.append((t, f))
        else:
            for v in node.values():
                out.extend(collect_pairs(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(collect_pairs(v))
    return out

# ---------------- Deterministic scorer ----------------

def score_field(field: str, query: str, kw_map: Dict[str, List[str]]) -> float:
    q = query.lower()
    f = field.lower()
    score = 0.0
    if f in q:
        score += 2.0
    for key, words in kw_map.items():
        if key in f:
            score += 0.6
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", q):
                score += 0.5
    # generic hints
    if any(phrase in q for phrase in ["based in", "located in", "address"]):
        score += 0.3
    return score

def deterministic_rank(query: str, allowed: List[Tuple[str, str]], keywords: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
    ranked = [(t, f, score_field(f, query, keywords)) for (t, f) in allowed]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked

# ---------------- Qwen (optional) ----------------

@dataclass
class QwenConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    temperature: float = 0.2
    top_p: float = 0.9

class QwenClient:
    def __init__(self, cfg: QwenConfig):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False
        )
        self.cfg = cfg

    def ask_json(self, prompt: str) -> dict:
        out = self.llm(
            prompt,
            max_tokens=768,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            stop=["</s>"]
        )
        text = out["choices"][0]["text"].strip()
        # Extract JSON if wrapped
        m = re.search(r"\{[\s\S]*\}$", text)
        if m: text = m.group(0)
        return json.loads(text)

def build_qwen_prompt(query: str, allowed: List[Tuple[str, str]], intent: str) -> str:
    allowed_str = "\n".join(f"- {t}.{f}" for (t, f) in allowed)
    return f"""You are a strict field selector for intent "{intent}".
Choose ONLY from the allowed fields.

ALLOWED FIELDS:
{allowed_str}

Rules:
- Output STRICT JSON only.
- JSON schema:
  {{
    "query": "<original>",
    "intent": "{intent}",
    "ranked_fields": [
      {{"table": "TableName", "field": "FieldName", "reason": "short reason", "confidence": 0.0}}
    ]
  }}
- Do not invent tables/fields.
- Order best→worst.

User query: "{query}"
Return the JSON now.
"""

def validate_model_json(model_json: dict, allowed: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    allowed_set = set(allowed)
    ranked: List[Tuple[str, str, float]] = []
    try:
        for it in model_json.get("ranked_fields", []):
            t = it.get("table"); f = it.get("field")
            conf = float(it.get("confidence", 0.0))
            if t and f and (t, f) in allowed_set:
                ranked.append((t, f, conf))
    except Exception:
        return []
    return ranked

# ---------------- The robust model ----------------

@dataclass
class IntentSpec:
    name: str
    groups: List[List[str]]          # paths into domain_mapping.json
    keywords: Dict[str, List[str]]   # scorer hints
    summary_template: Optional[str] = None

class IntentDemystifier:
    """
    Create once, reuse:
      model = IntentDemistifier(dm_path, intent_cfg_path)
      fields = model.select_fields(query, intent="location", top_k=6, qwen_cfg=...)
    """
    def __init__(self, domain_map_path: str | Path, intent_cfg_path: str | Path):
        self.dm_path = str(domain_map_path)
        self.intents = self._load_intents(intent_cfg_path)

        self.dm = load_json(self.dm_path)

    def _load_intents(self, path: str | Path) -> Dict[str, IntentSpec]:
        cfg = load_json(path)
        intents: Dict[str, IntentSpec] = {}
        for name, body in cfg.items():
            intents[name] = IntentSpec(
                name=name,
                groups=body.get("groups", []),
                keywords=body.get("keywords", {}),
                summary_template=(body.get("templates") or {}).get("summary")
            )
        return intents

    def _allowed_from_groups(self, spec: IntentSpec) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for gpath in spec.groups:
            node = get_node(self.dm, gpath)
            if node is None:
                continue
            pairs.extend(collect_pairs(node))
        # unique, stable order
        seen, uniq = set(), []
        for p in pairs:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq

    def select_fields(
        self,
        query: str,
        intent: str,
        top_k: int = 6,
        qwen_cfg: Optional[QwenConfig] = None
    ) -> List[Tuple[str, str]]:
        if intent not in self.intents:
            return []
        spec = self.intents[intent]
        allowed = self._allowed_from_groups(spec)
        if not allowed:
            return []

        # Try Qwen (if provided)
        if qwen_cfg is not None:
            try:
                client = QwenClient(qwen_cfg)
                prompt = build_qwen_prompt(query, allowed, intent)
                mj = client.ask_json(prompt)
                ranked = validate_model_json(mj, allowed)
                if ranked:
                    return [(t, f) for (t, f, _) in ranked[:top_k]]
            except Exception:
                pass  # fall back

        # Deterministic fallback
        ranked = deterministic_rank(query, allowed, spec.keywords)
        return [(t, f) for (t, f, _) in ranked[:top_k]]

    def summarize(self, record: Dict[str, Any], intent: str) -> Optional[str]:
        """Optional: fill a summary template using available fields."""
        spec = self.intents.get(intent)
        if not spec or not spec.summary_template:
            return None
        tmpl = spec.summary_template
        # naive formatter; missing keys become empty
        def getv(k: str) -> str:
            return str(record.get(k, "") or "")
        return re.sub(r"\{([^}]+)\}", lambda m: getv(m.group(1)), tmpl)
