# app_ask_enhanced.py
from flask import Flask, request, jsonify, Response, render_template, stream_with_context, send_from_directory, abort
from flask_cors import CORS
import os, json, time, re, random
from typing import Dict, Any, List, Optional
from pathlib import Path

# ---- Import the enhanced query engine ----
try:
    from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
except Exception:
    from enhanced_query_with_summary import execute_enhanced_query_with_summary  # type: ignore

# ----------------------------- Paths & Config -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIRS = [BASE_DIR / "templates", BASE_DIR]
STATIC_DIRS   = [BASE_DIR / "static", BASE_DIR]

def _read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _norm_path(raw: str, base: Path) -> Path:
    # Normalize slashes and resolve relative to `base`
    raw = raw.replace("\\", os.sep).replace("/", os.sep)
    pp = Path(raw)
    return (base / pp).resolve() if not pp.is_absolute() else pp.resolve()

# Load config.json (or the one at APP_CONFIG_PATH), then resolve company data dir
APP_CONFIG_PATH = Path(os.environ.get("APP_CONFIG_PATH", str(BASE_DIR / "config.json"))).resolve()
CFG = _read_json(APP_CONFIG_PATH) or {}

def _resolve_company_data_dir() -> Path:
    # 1) Highest priority: env var
    env_dir = os.environ.get("COMPANY_DATA_DIR")
    if env_dir:
        return _norm_path(env_dir, BASE_DIR)

    # 2) From config.json → company_mapped_data.processed_data_store
    #    Example in your config: "processed_data_store\\company_mapped_store"  :contentReference[oaicite:1]{index=1}
    try:
        cmd = CFG.get("company_mapped_data") or {}
        ps  = cmd.get("processed_data_store")
        if ps:
            return _norm_path(ps, BASE_DIR)
    except Exception:
        pass

    # 3) Fallback: ./processed_data_store/company_mapped_store
    return (BASE_DIR / "processed_data_store" / "company_mapped_store").resolve()

COMPANY_DATA_DIR = _resolve_company_data_dir()
DEFAULT_TOPK     = int(os.environ.get("ASKAPP_TOPK", "30"))
MAX_ROWS_PAYLOAD = int(os.environ.get("ASKAPP_MAX_ROWS", "20"))

# ---------------------- Small helpers -----------------------------
_GREET_RE  = re.compile(r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|namaste)\b", re.I)
_THANKS_RE = re.compile(r"\b(thanks|thank\s*you|much\s*appreciated)\b", re.I)
_BYE_RE    = re.compile(r"\b(bye|goodbye|see\s*you|ttyl|take\s*care)\b", re.I)
_HELP_RE   = re.compile(r"\b(help|how\s*(do|to)|what\s*can\s*you\s*do|examples?)\b", re.I)
_OFFTOPIC_PATTERNS = [
    r"\bweather\b", r"\btemperature\b", r"\brain\b",
    r"\b(date|time)\b", r"\bday\s*is\s*it\b",
    r"\bnews\b", r"\bcricket\b", r"\bfootball\b", r"\bscore\b",
    r"\bmovie\b", r"\bfilm\b", r"\bcelebrity\b",
    r"\bjoke\b", r"\briddle\b", r"\bpoem\b", r"\bstory\b",
    r"\bstock\b", r"\bbitcoin\b", r"\bexchange\s*rate\b"
]
_OFFTOPIC_RE = re.compile("|".join(_OFFTOPIC_PATTERNS), re.I)

_GREETINGS = [
    "Hello! 👋 How can I help you with companies, products, or certifications?",
    "Hi! Ask me about companies (location, domain), ISO certifications, products, or revenue.",
    "Namaste! You can search by certification (e.g., ISO 9001), product (e.g., High Voltage Transformer), or turnover."
]
_HELP_TEXT = (
    "You can ask things like:\n"
    "• \"ISO 9001 certified companies in Karnataka\"\n"
    "• \"Companies with High Voltage Transformer products\"\n"
    "• \"Revenue / turnover details for aerospace vendors\"\n"
    "• \"List companies in Chennai with ISO/IEC 27001\"\n"
)

def classify_intent(text: str) -> str:
    t = (text or "").strip()
    if not t: return "unknown"
    if _GREET_RE.search(t):  return "greet"
    if _THANKS_RE.search(t): return "thanks"
    if _BYE_RE.search(t):    return "bye"
    if _HELP_RE.search(t):   return "help"
    if _OFFTOPIC_RE.search(t): return "offtopic"
    return "unknown"

def quick_reply(intent: str) -> str:
    if intent == "greet":  return random.choice(_GREETINGS)
    if intent == "thanks": return "You're welcome!"
    if intent == "bye":    return "Goodbye! 👋"
    if intent == "help":   return _HELP_TEXT
    if intent == "offtopic":
        return ("I'm focused on the company/product search (certifications, domains, products, revenue). "
                "Try: ISO certified companies in India; vendors for High Voltage Transformer; "
                "turnover details for electronics domain.")
    return ""

def sanitize(text: str) -> str:
    out = text or ""
    for pat in [r"<<SYS>>.*?<</SYS>>", r"<>.*?<>"]:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out.strip()

def summarize_markdown(result: Dict[str, Any]) -> str:
    companies = result.get("companies") or []
    summary = (result.get("enhanced_summary") or "").strip()
    intent_answer = (result.get("intent_answer") or "").strip()
    parts = []
    if summary: parts.append(summary)
    if intent_answer: parts.append(intent_answer)
    if companies:
        rows = []
        for c in companies[:5]:
            # Show only essential fields: Company Name, Ref No, Website, Email, Phone
            website = c.get('website', '') or ''
            email = c.get('email', '') or ''
            phone = c.get('phone', '') or ''
            rows.append(f"| {c.get('company_name','-')} | {c.get('company_ref_no','-')} | "
                        f"{website} | {email} | {phone} |")
        if rows:
            parts.append("\n**Top matches**\n\n| Company | Ref No | Website | Email | Phone |\n|---|---|---|---|---|\n" + "\n".join(rows))
    return sanitize("\n\n".join([p for p in parts if p])) or "No matching companies found."

def trim_rows_for_ui(result: Dict[str, Any], max_rows: int = MAX_ROWS_PAYLOAD) -> List[Dict[str, Any]]:
    companies = result.get("companies") or []
    view = []
    for c in companies[:max_rows]:
        # Only include essential fields: Company Name, Ref No, Website, Email, Phone
        row = {
            "CompanyName": c.get("company_name"),
            "CompanyRefNo": c.get("company_ref_no"),
        }
        # Add contact fields only if they have values
        if c.get("website"):
            row["Website"] = c.get("website")
        if c.get("email"):
            row["Email"] = c.get("email")
        if c.get("phone"):
            row["Phone"] = c.get("phone")
        view.append(row)
    return view

# ------------------------------ Flask app -----------------------------------
app = Flask(__name__, template_folder=str(TEMPLATE_DIRS[0]), static_folder=str(STATIC_DIRS[0]))
CORS(app)

def _find_in(dirs: List[Path], filename: str) -> Path:
    for d in dirs:
        p = d / filename
        if p.exists(): return p
    return Path()

@app.route("/")
def home():
    p = _find_in(TEMPLATE_DIRS, "portal.html")
    if p.exists():
        if str(p.parent) == app.template_folder:
            return render_template("portal.html")
        return send_from_directory(p.parent, p.name)
    return "Ask app is running. Put portal.html into ./templates or this folder."

@app.route("/chat")
def chat_page():
    p = _find_in(TEMPLATE_DIRS, "index.html")
    if p.exists():
        if str(p.parent) == app.template_folder:
            return render_template("index.html")
        return send_from_directory(p.parent, p.name)
    abort(404, description="index.html not found. Place it under ./templates or alongside the app.")

@app.route("/static/<path:filename>")
def static_fallback(filename: str):
    try:
        return app.send_static_file(filename)
    except Exception:
        pass
    p = _find_in(STATIC_DIRS, filename)
    if p.exists():
        return send_from_directory(p.parent, p.name)
    if filename == "askme-modal.js":
        p2 = BASE_DIR / "askme-modal.js"
        if p2.exists():
            return send_from_directory(p2.parent, p2.name)
    abort(404)

# --------------- Non-streaming: JSON in / JSON out --------------------------
@app.route("/ask", methods=["POST"])
def ask():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    intent = classify_intent(query)
    if intent in {"greet", "thanks", "bye", "help", "offtopic"}:
        return jsonify({
            "status": "success",
            "answer": quick_reply(intent),
            "rows": [],
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        })

    try:
        # Pass both styles so the engine can use either:
        #  - company_data (folder containing integrated_company_search.json)
        #  - company_mapped_data.processed_data_store (as in your config)
        result = execute_enhanced_query_with_summary(
            user_query=query,
            config={
                "company_data": str(COMPANY_DATA_DIR),
                "company_mapped_data": {"processed_data_store": str(COMPANY_DATA_DIR)}
            },
            logger=None,
            topk=DEFAULT_TOPK
        )
        answer = summarize_markdown(result)
        rows = trim_rows_for_ui(result, max_rows=MAX_ROWS_PAYLOAD)
        return jsonify({
            "status": "success",
            "answer": answer,
            "rows": rows,
            "metadata": result.get("metadata", {}),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- SSE streaming version -------------------------------------
@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    intent = classify_intent(query)
    if intent in {"greet", "thanks", "bye", "help", "offtopic"}:
        def quick_gen():
            yield "event: status\ndata: quick\n\n"
            msg = quick_reply(intent)
            yield f"event: token\ndata: {json.dumps(msg)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return Response(stream_with_context(quick_gen()), mimetype="text/event-stream")

    @stream_with_context
    def generate():
        try:
            yield "event: status\ndata: retrieving\n\n"
            result = execute_enhanced_query_with_summary(
                user_query=query,
                config={
                    "company_data": str(COMPANY_DATA_DIR),
                    "company_mapped_data": {"processed_data_store": str(COMPANY_DATA_DIR)}
                },
                logger=None,
                topk=DEFAULT_TOPK
            )
            yield "event: status\ndata: generating\n\n"
            final_text = summarize_markdown(result)
            CHUNK = 140
            for i in range(0, len(final_text), CHUNK):
                chunk = final_text[i:i+CHUNK]
                if chunk:
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/healthz")
def healthz():
    exists = (COMPANY_DATA_DIR / "integrated_company_search.json").exists()
    return jsonify({
        "ok": True,
        "config_path": str(APP_CONFIG_PATH),
        "company_data_dir": str(COMPANY_DATA_DIR),
        "json_present": exists
    })

if __name__ == "__main__":
    if os.name == "nt":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting Ask app on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
