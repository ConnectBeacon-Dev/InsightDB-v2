# app.py
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import json, time, re, random
from datetime import datetime
import pandas as pd

# === Import your pipeline ONCE; everything stays in memory ===
# The module should define: process_query, run_llm_mappings, and a llama_cpp Llama instance `llm`
from build_query_domain_final import process_query, run_llm_mappings, llm, summarize_with_llm_timeout
from load_config import load_config

# Load config for paths
config = load_config()

# --------- Output/Speed knobs (tune as you like) ----------
MAX_SUMMARY_TOKENS = 250     # keep small on CPU
MAX_SUMMARY_ROWS   = 8       # preview top rows to LLM only
DEFAULT_TOP_K      = 30      # was 100
DEFAULT_MIN_SCORE  = 0.50    # was 0.35

SYSTEM_MSG = (
    "You format enterprise search results for business users. Be concise. "
    "Prefer short bullet points and small markdown tables when useful. "
    "Do not repeat these instructions in your answer."
)

def build_user_msg(query, lookups, df_head_text):
    return (
        f"Query: {query}\n"
        f"Lookups: {lookups}\n"
        f"Rows (preview):\n{df_head_text}\n\n"
        "Write the answer in clear markdown. Do not include any system/instruction text."
    )

# Safety net: strip any echoed instructions, sys wrappers, etc.
ECHO_PATTERNS = [
    r"You format enterprise search results.*",
    r"<<SYS>>.*?<</SYS>>",
    r"<>.*?<>",
]
def sanitize(text: str) -> str:
    out = text or ""
    for pat in ECHO_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out.strip()

# ---------- FAST INTENT LAYER (no LLM) ----------
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
    "Hello! 👋 How can I help you with defense products or companies?",
    "Hi there! Ask me about companies, certifications (e.g., ISO 9001), platforms, or expertise.",
    "Namaste! You can search by URN, certifications, country, sector, etc."
]
_HELP_TEXT = (
    "I'm specialized in the Defense Product Search.\n\n"
    "You can ask things like:\n"
    "• \"ISO 9001 certified companies in India\"\n"
    "• \"Aerospace domain vendors with R&D in electrical\"\n"
    "• \"List companies having products in defence sector\"\n"
    "• \"Companies by URN = PE00032\"\n\n"
    "I'll fetch matching rows and summarize them for you."
)

def classify_intent(text: str) -> str:
    """Return one of: greet, thanks, bye, help, offtopic, unknown."""
    t = (text or "").strip()
    if not t:
        return "unknown"
    if _GREET_RE.search(t):  return "greet"
    if _THANKS_RE.search(t): return "thanks"
    if _BYE_RE.search(t):    return "bye"
    if _HELP_RE.search(t):   return "help"
    if _OFFTOPIC_RE.search(t): return "offtopic"
    return "unknown"

def quick_reply(intent: str) -> str:
    if intent == "greet":
        return random.choice(_GREETINGS)
    if intent == "thanks":
        return "You're welcome! If you need anything else, ask away."
    if intent == "bye":
        return "Goodbye! 👋"
    if intent == "help":
        return _HELP_TEXT
    if intent == "offtopic":
        return (
            "I'm focused on the Defense Product Search (companies, products, certifications, domains, R&D, URN, etc.).\n"
            "Please ask a product/company question—for example:\n"
            "• ISO/NABL certified companies in India\n"
            "• Companies in aerospace domain with electrical R&D\n"
            "• Vendors for a specific defense platform\n"
        )
    return ""

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Optional: warm caches so first request is faster
def warmup():
    try:
        _ = process_query("warmup")
        df = pd.DataFrame([{"CompanyRefNo": "X", "CompanyNumber": "Y"}])
        _ = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_user_msg("warmup", [], df.head(1).to_string(index=False))}
            ],
            temperature=0.2,
            max_tokens=16,
        )
        print("[warmup] OK")
    except Exception as e:
        print("[warmup] skipped:", e)

# --------- Pages ---------
@app.route("/")
def home():
    # You can point your portal to /portal; this keeps / available if needed
    return render_template("portal.html")

@app.route("/portal")
def portal():
    return render_template("portal.html")

@app.route("/chat")
def chat():
    # If your chat page is named templates/index.html, keep this.
    # If you renamed it to chat.html, change to render_template("chat.html")
    return render_template("index.html")

# --------- API: non-streaming ----------
@app.route("/ask", methods=["POST"])
def ask():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    # 0) Fast intent handling
    intent = classify_intent(query)
    if intent in {"greet", "thanks", "bye", "help", "offtopic"}:
        return jsonify({
            "status": "success",
            "answer": quick_reply(intent),
            "rows": [],
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        })

    try:
        # 1) Retrieval (your pipeline)
        t1 = time.perf_counter()
        llm_out = process_query(query)
        # Use union directly like the working build_query_domain_final.py
        df = run_llm_mappings(llm_out, 
            stores_root=config.get("generated_embedding_store"),
            top_k=DEFAULT_TOP_K,
            min_score=DEFAULT_MIN_SCORE,
            prefer_exact=True,
            boost_exact=0.20,
            combine="union",   # Use union directly for consistency with working version
            columns=["_id","_score","CompanyRefNo","CompanyNumber","Domain","_doc"]
            )
        
        combine_note = None
        
        lookups = [(d.get("domain"), d.get("subdomain"), d.get("value")) for d in llm_out if isinstance(d, dict)]
        t2 = time.perf_counter()

        # Use optimized fast summarization with pre-generated company info
        summary = summarize_with_llm_timeout(
            df,
            query=query,
            lookups=lookups,
            llm=llm,
            max_rows=MAX_SUMMARY_ROWS,
            temperature=0.2,
            max_tokens=MAX_SUMMARY_TOKENS,
            timeout=30,  # Fast timeout since we're using local enhanced summarization
            base_path=config.get("domain_mapped_csv_store")
        ) if df is not None else "No matching rows."
       
        if combine_note and summary:
            summary = f"{summary}\n\n_{combine_note}_"
                
        # 3) Chat completion (prevents instruction echo)
        rows_payload = []
        if df is not None and not df.empty:
            # choose a few helpful columns for the UI table (don't limit DF itself)
            cols = [c for c in df.columns if not c.startswith("_")]
            if "CompanyRefNo" in cols: pass
            elif "CompanyMaster.CompanyRefNo" in cols: cols.insert(0, "CompanyMaster.CompanyRefNo")
            rows_payload = df[cols[:8]].head(20).to_dict(orient="records")

        t3 = time.perf_counter()
        print(f"[timing] total={t3-t0:.2f}s | retrieval={t2-t1:.2f}s | summarize={t3-t2:.2f}s")

        return jsonify({"status": "success", "answer": summary, "rows": rows_payload})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --------- API: streaming (SSE over POST) ----------
@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    # 0) Quick intents -> stream one-shot reply
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
            # -------- Phase 1: retrieval (strict first) --------
            t1 = time.perf_counter()
            yield "event: status\ndata: retrieving\n\n"

            llm_out = process_query(query)

            df = run_llm_mappings(
                llm_out,
                stores_root=config.get("generated_embedding_store"),
                top_k=DEFAULT_TOP_K,
                min_score=DEFAULT_MIN_SCORE,
                prefer_exact=True,
                boost_exact=0.20,
                combine="union",  # Use union for consistency with working version
                columns=["_id","_score","CompanyRefNo","CompanyNumber","Domain","_doc"]  # optional view
            )
                        
            lookups = [
                (d.get("domain"), d.get("subdomain"), d.get("value"))
                for d in llm_out if isinstance(d, dict)
            ]

            # -------- Phase 2: summarize with optimized fast helper --------
            yield "event: status\ndata: generating\n\n"

            if df is None:
                final_text = "No matching rows."
            else:
                # Use optimized fast summarization with pre-generated company info
                final_text = summarize_with_llm_timeout(
                    df,
                    query=query,
                    lookups=lookups,
                    llm=llm,
                    max_rows=MAX_SUMMARY_ROWS,
                    temperature=0.2,
                    max_tokens=MAX_SUMMARY_TOKENS,
                    timeout=30,  # Fast timeout since we're using local enhanced summarization
                    base_path=config.get("domain_mapped_csv_store")
                )
                       
            # -------- Phase 3: stream the summary text as tokens --------
            # We chunk by ~120 characters to keep updates smooth
            CHUNK_SIZE = 120
            i = 0
            n = len(final_text)
            while i < n:
                chunk = final_text[i:i+CHUNK_SIZE]
                i += CHUNK_SIZE
                if chunk:
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    import os
    import sys
    
    # Fix Windows console encoding issues
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        # Suppress Flask banner to avoid Windows console issues
        os.environ['FLASK_SKIP_DOTENV'] = '1'
    
    warmup()
    print("Starting Flask server on http://0.0.0.0:8000")
    
    # IMPORTANT: debug=False to avoid the reloader (keeps single process, single model in RAM)
    try:
        app.run(host="0.0.0.0", port=8000, debug=False)
    except OSError as e:
        if "Windows error 6" in str(e):
            print("Note: Windows console output error occurred, but server should be running normally.")
            print("You can access the application at http://localhost:8000")
        else:
            raise
