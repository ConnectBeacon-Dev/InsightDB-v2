# summarize_result_enhanced.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
import os
from pathlib import Path

try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # type: ignore


def _ensure_domain_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Domain" in df.columns:
        return df
    if "Group" in df.columns:
        g = df["Group"].fillna("").astype(str)
        s = df["Subgroup"].fillna("").astype(str) if "Subgroup" in df.columns else ""
        df = df.copy()
        df["Domain"] = (g + "." + s).str.strip(".")
    return df


def _highlight(text: str, terms: List[str], max_chars: int = 160) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t_lower = text.lower().replace("\\n", " ").replace("\n", " ")
    hits = [(t_lower.find(term.lower()), term) for term in terms if term]
    hits = [h for h in hits if h[0] >= 0]
    if not hits:
        return (t_lower[:max_chars] + "…") if len(t_lower) > max_chars else t_lower
    pos, _ = sorted(hits, key=lambda x: x[0])[0]
    start = max(0, pos - 60)
    end = min(len(t_lower), pos + 100)
    snippet = t_lower[start:end]
    return snippet if len(snippet) <= max_chars else (snippet[:max_chars] + "…")


def load_company_summary(company_ref_no: str, base_path) -> Optional[Dict]:
    """Load pre-generated company summary from company_mapped_store directory"""
    try:
        # Try the direct path first (our current structure)
        info_file = Path(base_path) / f"{company_ref_no}_INFO.json"
        
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Fallback to the old structure if needed
        info_dir = Path(base_path) / f"{company_ref_no}_information"
        info_file_old = info_dir / f"{company_ref_no}_INFO.json"
        
        if info_file_old.exists():
            with open(info_file_old, 'r', encoding='utf-8') as f:
                return json.load(f)
                
    except Exception as e:
        # Silently fail and return None
        pass
    return None


def extract_company_details(company_info: Dict) -> Dict[str, str]:
    """Extract key details from company info for summary"""
    details = {
        'name': '',
        'location': '',
        'industry': '',
        'expertise': '',
        'scale': '',
        'status': '',
        'summary': '',
        'expanded_summary': ''
    }
    
    try:
        # Get LLM generated summary if available
        llm_summary = company_info.get('llm_generated_summary', {})
        details['summary'] = llm_summary.get('summary', '')
        details['expanded_summary'] = llm_summary.get('expanded', '')
        
        # Extract from complete data
        complete_data = company_info.get('complete_data', {})
        
        # Basic info
        basic_info = complete_data.get('CompanyProfile.BasicInfo', {}).get('data', [])
        if basic_info:
            basic = basic_info[0]
            details['name'] = basic.get('CompanyMaster.CompanyName', '')
            details['status'] = basic.get('CompanyMaster.CompanyStatus', '')
        
        # Contact info for location
        contact_info = complete_data.get('CompanyProfile.ContactInfo', {}).get('data', [])
        if contact_info:
            contact = contact_info[0]
            city = contact.get('CompanyMaster.CityName', '')
            state = contact.get('CompanyMaster.State', '')
            country = contact.get('CountryMaster.CountryName', '')
            details['location'] = f"{city}, {state}, {country}".strip(', ')
        
        # Classification
        classification = complete_data.get('CompanyProfile.Classification', {}).get('data', [])
        if classification:
            classif = classification[0]
            details['scale'] = classif.get('ScaleMaster.CompanyScale', '')
        
        # Industry
        industry_info = complete_data.get('BusinessDomain.Industry', {}).get('data', [])
        if industry_info:
            industry = industry_info[0]
            domain_type = industry.get('IndustryDomainMaster.IndustryDomainType', '')
            subdomain = industry.get('IndustrySubdomainType.IndustrySubDomainName', '')
            details['industry'] = f"{domain_type} - {subdomain}".strip(' - ')
        
        # Expertise
        expertise_info = complete_data.get('BusinessDomain.CoreExpertise', {}).get('data', [])
        if expertise_info:
            expertise = expertise_info[0]
            details['expertise'] = expertise.get('CompanyCoreExpertiseMaster.CoreExpertiseName', '')
            
    except Exception:
        pass
    
    return details


def _df_to_payload_enhanced(
    df: pd.DataFrame,
    *,
    lookups: List[Tuple[str, str, str]] | None = None,
    top_k_rows: int = 20,
    include_score: bool = True,
    base_path
) -> Dict:
    """Enhanced payload generation using pre-generated company summaries"""
    df = _ensure_domain_col(df)
    cols = [c for c in ["CompanyRefNo", "CompanyNumber", "CompanyName", "Domain", "_score", "_doc", "__store", "__value"] if c in df.columns]
    small = df[cols].head(top_k_rows).copy()

    terms = [v for (_, _, v) in (lookups or []) if v]

    rows = []
    for _, r in small.iterrows():
        company_ref_no = r.get("CompanyRefNo", "")
        
        # Try to load pre-generated company summary
        company_info = load_company_summary(company_ref_no, base_path) if company_ref_no else None
        
        if company_info:
            # Use pre-generated summary
            details = extract_company_details(company_info)
            reason = details.get('summary') or details.get('expanded_summary') or _highlight(str(r.get("_doc", "")), terms=terms, max_chars=160)
            company_name = details.get('name') or r.get("CompanyName", "")
        else:
            # Fallback to original method
            reason = _highlight(str(r.get("_doc", "")), terms=terms, max_chars=160)
            company_name = r.get("CompanyName", "")

        rows.append({
            "company_ref_no": company_ref_no,
            "company_no": r.get("CompanyNumber", ""),
            "name": company_name,
            "domain": r.get("Domain", r.get("__store", "")),
            "score": float(r["_score"]) if include_score and "_score" in r and pd.notna(r["_score"]) else None,
            "reason": reason,
            "has_summary": company_info is not None
        })

    payload = {
        "meta": {
            "total_rows": int(len(df)),
            "shown_rows": int(len(rows)),
            "distinct_companies": int(len(df.drop_duplicates(subset=[c for c in ["CompanyRefNo", "CompanyNumber"] if c in df.columns]))),
            "companies_with_summaries": sum(1 for row in rows if row.get("has_summary", False)),
            "lookups": [{"domain": d, "subdomain": s, "value": v} for (d, s, v) in (lookups or [])],
        },
        "rows": rows,
    }
    return payload


# --------- ENHANCED PROMPTS ---------
ENHANCED_SYSTEM = (
    "You format enterprise search results for business users using pre-generated company summaries. "
    "Do NOT use markdown, tables, bullet points, or code fences. "
    "Write a concise plain-text summary of 2-4 sentences, then list company details. "
    "Use the provided company summaries when available for richer context."
)

ENHANCED_USER_TEMPLATE = """Summarize the following result set for the user query using available company summaries.

Query:
{query}

Constraints (Domain.Subdomain = value):
{constraints}

Data (JSON with enhanced company summaries):
{payload}

OUTPUT FORMAT (strict):
1) First, write a plain-text summary of 2 to 4 sentences describing:
   - How many companies matched and how many have detailed summaries
   - What they have in common based on the query constraints
   - Key insights from the company summaries when available
   
2) Then write a 'Companies:' section (exactly the word Companies: on a single line), followed by up to {max_rows} company entries. Each entry should be:
URN=<CompanyRefNo> | Name=<CompanyName> | Domain=<Domain> | Summary=<reason/summary>
- Use the enhanced company summary when available (has_summary=true)
- Keep summaries concise but informative (under 200 characters)
- If no company name, leave it blank after Name=
- Do NOT use markdown. Do NOT output any extra headers or explanations.

Begin now.
"""


def summarize_with_enhanced_summaries(
    logger,
    df: pd.DataFrame,
    *,
    query: str,
    lookups: List[Tuple[str, str, str]] | None = None,
    llm: Optional["Llama"] = None,
    model_path: Optional[str] = None,
    n_threads: int = 8,
    n_ctx: int = 4096,
    max_rows: int = 15,
    temperature: float = 0.2,
    max_tokens: int = 800,  # Increased for richer summaries
    base_path
) -> str:
    """Enhanced summarization using pre-generated company summaries"""
    
    if df is None or df.empty:
        logger.debug("summarize_with_enhanced_summaries: empty dataframe")
        return "No matching companies were found for the given constraints.\nCompanies:"

    # Build enhanced payload with company summaries
    try:
        payload = _df_to_payload_enhanced(df, lookups=lookups, top_k_rows=max_rows, base_path=base_path)
        companies_with_summaries = payload["meta"]["companies_with_summaries"]
        logger.debug(f"Enhanced payload: {companies_with_summaries}/{len(payload['rows'])} companies have summaries")
    except Exception as e:
        logger.debug(f"Enhanced payload failed: {e}, falling back to local summarizer")
        return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)

    # If we have many companies with summaries, use enhanced LLM approach
    if companies_with_summaries > 0:
        constraints = "\n".join([f"- {d}.{s} = {v}" for (d, s, v) in (lookups or [])]) or "- (none)"
        
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.debug(f"JSON serialization failed: {e}")
            return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)

        user_msg = ENHANCED_USER_TEMPLATE.format(
            query=query,
            constraints=constraints,
            payload=payload_json,
            max_rows=max_rows
        )
        prompt = f"[INST]<<SYS>>{ENHANCED_SYSTEM}<<SYS>>\n{user_msg}\n[/INST]"

        # LLM processing
        local_llm = llm
        if local_llm is None:
            if model_path is None or Llama is None:
                logger.debug("No LLM available, using local enhanced summarizer")
                return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)
            try:
                local_llm = Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx)
            except Exception as e:
                logger.debug(f"LLM init failed: {e}")
                return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)

        # Generate summary with timeout protection
        import time
        t0 = time.perf_counter()
        try:
            resp = local_llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"]  # Additional stop tokens to prevent runaway generation
            )
            dt = time.perf_counter() - t0
            
            text = resp["choices"][0]["text"].strip()
            logger.debug(f"Enhanced LLM summary generated in {dt:.2f}s, length: {len(text)}")
            
            # Validate the output has the expected structure
            if "Companies:" in text or "Rows:" in text:
                return text
            else:
                logger.debug("LLM output missing expected structure, using local fallback")
                return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)
                
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.debug(f"LLM completion failed in {dt:.2f}s: {e}")
            return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)
    
    # Fallback to local enhanced summarizer
    return summarize_locally_enhanced(df, query=query, lookups=lookups, max_rows=max_rows, base_path=base_path)


def summarize_locally_enhanced(
    df: pd.DataFrame,
    *,
    query: str,
    lookups: List[Tuple[str, str, str]] | None = None,
    max_rows: int = 15,
    base_path: str = "domain_mapped_csv_store"
) -> str:
    """Enhanced local summarization using pre-generated company summaries with improved formatting"""
    df = _ensure_domain_col(df)
    total = len(df)
    distinct = len(df.drop_duplicates(subset=[c for c in ["CompanyRefNo", "CompanyNumber"] if c in df.columns]))
    constraints = "; ".join([f"{d}.{s}={v}" for (d, s, v) in (lookups or [])]) or "(none)"

    lines: List[str] = []
    
    # Count companies with summaries
    companies_with_summaries = 0
    shown = df.head(max_rows).copy()
    
    for _, r in shown.iterrows():
        company_ref_no = r.get("CompanyRefNo", "")
        if company_ref_no and load_company_summary(company_ref_no, base_path):
            companies_with_summaries += 1

    # Enhanced summary with better formatting
    lines.append(f"QUERY RESULTS")
    lines.append(f"Query: {query}")
    lines.append("")
    lines.append(f"📊 SUMMARY:")
    lines.append(f"   • Found {total} matches across {distinct} companies")
    lines.append(f"   • Constraints: {constraints}")
    if companies_with_summaries > 0:
        lines.append(f"   • Enhanced details available for {companies_with_summaries} companies")
    lines.append("")
    
    if total == 0:
        lines.append("❌ No matching companies were found.")
        return "\n".join(lines)

    lines.append(f"📋 SHOWING TOP {min(max_rows, total)} RESULTS:")
    
    # Build enhanced company entries with better formatting
    terms = [v for (_, _, v) in (lookups or []) if v]
    for i, (_, r) in enumerate(shown.iterrows(), 1):
        company_ref_no = r.get("CompanyRefNo", "")
        company_info = load_company_summary(company_ref_no, base_path) if company_ref_no else None
        
        if company_info:
            # Use enhanced company details
            details = extract_company_details(company_info)
            summary = details.get('expanded_summary') or details.get('summary')
            if not summary:
                summary = _highlight(str(r.get("_doc", "")), terms=terms, max_chars=200)
            company_name = details.get('name') or r.get("CompanyName", "")
        else:
            # Fallback to basic highlighting
            summary = _highlight(str(r.get("_doc", "")), terms=terms, max_chars=200)
            company_name = r.get("CompanyName", "")
        
        lines.append(f"🏢 COMPANY #{i}")
        lines.append(f"   URN Number: {company_ref_no or 'N/A'}")
        lines.append(f"   Name: {company_name or 'N/A'}")
        lines.append(f"   Domain: {str(r.get('Domain', '') or 'N/A')}")
        lines.append(f"   Summary: {summary or 'No details available'}")
        if i < len(shown):
            lines.append("")  # Add spacing between companies
    
    return "\n".join(lines)


# Backward compatibility functions
def summarize_with_llm(
    df: pd.DataFrame,
    query: str,
    lookups: List[Tuple[str, str, str]] | None = None,
    llm: Optional["Llama"] = None,
    model_path: Optional[str] = None,
    n_threads: int = 8,
    n_ctx: int = 4096,
    max_rows: int = 15,
    temperature: float = 0.2,
    max_tokens: int = 600
) -> str:
    """Backward compatible function that uses enhanced summarization"""
    return summarize_with_enhanced_summaries(
        df,
        query=query,
        lookups=lookups,
        llm=llm,
        model_path=model_path,
        n_threads=n_threads,
        n_ctx=n_ctx,
        max_rows=max_rows,
        temperature=temperature,
        max_tokens=max_tokens
    )


def summarize_locally(
    df: pd.DataFrame,
    *,
    query: str,
    lookups: List[Tuple[str, str, str]] | None = None,
    max_rows: int = 15
) -> str:
    """Backward compatible function that uses enhanced local summarization"""
    return summarize_locally_enhanced(
        df,
        query=query,
        lookups=lookups,
        max_rows=max_rows
    )
