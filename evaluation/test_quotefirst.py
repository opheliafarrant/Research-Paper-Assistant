"""
Quote-First Generation and Verbatim Verification Audit Suite
Formalizes the two-stage Quote-First generation pipeline and performs physical verbatim verification:
1. Stage 1: Extract verbatim quotes with [Chunk ID].
2. Verbatim Check: Strips '...' and checks exact substring presence in raw chunk text.
3. Stage 2: Chinese synthesis strictly grounded in extracted quotes.
4. Fact & Omission Audit: Evaluates whether Chinese claims are supported and detects omitted ground-truth facts.
"""

import sys
import os
import re
import json
import difflib
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from research_assistant.infrastructure.indexes import load_indexes
from research_assistant.infrastructure.embedding import embed
from research_assistant.application.aspects import get_aspect_strategy, retrieve_aspect_chunks
from research_assistant.domain.contracts import ResearchDependencies
from research_assistant.infrastructure.ollama import generate


def clean_quote_for_matching(quote_text: str) -> str:
    """Strip ellipses, markdown quotes, and leading/trailing whitespace."""
    cleaned = quote_text.strip()
    # Strip markdown blockquote symbols
    cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
    # Strip leading/trailing quotation marks
    cleaned = cleaned.strip('"\'“”‘’')
    # Strip ellipses
    cleaned = re.sub(r"\.\.\.", " ", cleaned)
    cleaned = re.sub(r"…", " ", cleaned)
    # Collapse multiple whitespaces/newlines to single space for normalized comparison
    return cleaned.strip()


def verify_verbatim_presence(quote: str, chunk_text: str) -> dict:
    """
    Check if quote exists verbatim in chunk_text.
    Checks both strict exact substring and whitespace-normalized substring.
    """
    cleaned_quote = clean_quote_for_matching(quote)
    
    # Check 1: Strict substring match
    if cleaned_quote in chunk_text:
        return {
            "verbatim_exact": True,
            "match_ratio": 1.0,
            "method": "strict_substring"
        }
    
    # Check 2: Normalized whitespace substring match
    norm_quote = " ".join(cleaned_quote.split())
    norm_chunk = " ".join(chunk_text.split())
    if norm_quote in norm_chunk:
        return {
            "verbatim_exact": True,
            "match_ratio": 1.0,
            "method": "whitespace_normalized"
        }
    
    # Check 3: Fuzzy / diff ratio if not strictly verbatim
    matcher = difflib.SequenceMatcher(None, norm_quote, norm_chunk)
    match = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_chunk))
    ratio = matcher.quick_ratio()
    
    return {
        "verbatim_exact": False,
        "match_ratio": round(match.size / max(1, len(norm_quote)), 4),
        "longest_match_len": match.size,
        "quote_len": len(norm_quote),
        "method": "partial_mismatch"
    }


def parse_stage1_quotes(stage1_text: str) -> list[dict]:
    """
    Parse Stage 1 output into structured list of {chunk_idx, quote}.
    Handles patterns like:
    '[Chunk 128, Page 12]: "..."'
    '[Chunk 128]: "..."'
    """
    results = []
    # Match blocks by [Chunk <id> ...]
    pattern = re.compile(r'\[Chunk\s+(\d+)(?:[^\]]*)\]:?\s*\n*["“](.*?)["”]', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(stage1_text)
    for cid_str, quote_str in matches:
        cid = int(cid_str)
        quote = quote_str.strip()
        if len(quote) > 10:
            results.append({"chunk_idx": cid, "quote": quote})
            
    # Fallback if quotes were not enclosed in quotation marks
    if not results:
        blocks = re.split(r'\[Chunk\s+(\d+)(?:[^\]]*)\]', stage1_text)
        if len(blocks) > 1:
            for i in range(1, len(blocks), 2):
                cid = int(blocks[i])
                content = blocks[i+1].strip().strip(":").strip()
                cleaned_lines = [l.strip().lstrip('>"*- ') for l in content.split("\n") if len(l.strip()) > 20]
                if cleaned_lines:
                    results.append({"chunk_idx": cid, "quote": " ".join(cleaned_lines)})
    return results


def run_quotefirst_audit():
    print("=" * 80)
    print("QUOTE-FIRST TWO-STAGE GENERATION & VERBATIM AUDIT")
    print("=" * 80)

    # 1. Load ground truth
    with open("evaluation/limitations_cases.json", "r", encoding="utf-8") as f:
        cases_data = json.load(f)

    # We evaluate on Case 1 (2611.09892), which has detailed ground truth facts
    target_case = [c for c in cases_data["cases"] if c["paper_id"] == "2611.09892"][0]
    pid = target_case["paper_id"]
    ground_truth_facts = target_case["facts"]

    # 2. Retrieve aspect chunks using production strategy
    faiss_idx, bm25_idx = load_indexes()
    deps = ResearchDependencies(
        load_indexes=load_indexes,
        embed=embed,
        generate=lambda *args, **kwargs: ""
    )
    strategy = get_aspect_strategy("limitations")
    retrieved_chunks = retrieve_aspect_chunks(pid, title=target_case["paper_title"], strategy=strategy, dependencies=deps, top_k=12)

    retrieved_chunk_indices = [c["chunk_idx"] for c in retrieved_chunks]
    print(f"Retrieved Top-{len(retrieved_chunks)} Chunks for {pid}: {retrieved_chunk_indices}")

    context = "\n\n---\n\n".join(f"[Chunk {c['chunk_idx']}, Page {c['page']}]\n{c['text']}" for c in retrieved_chunks)

    # Chunk text lookup for physical verification
    chunk_lookup = {c["chunk_idx"]: c["text"] for c in retrieved_chunks}

    # 3. Stage 1: Verbatim Quote Extraction
    sys_prompt_q1 = (
        "You are a strict research assistant. Your task is to extract exact, verbatim quotes "
        "from the provided paper excerpts that discuss the limitations, failure modes, assumptions, "
        "or future directions of the work. For each quote, specify the [Chunk ID]. "
        "Do not summarize or paraphrase. If none, return empty."
    )
    prompt_q1 = f"Excerpts:\n{context}\n\nExtract verbatim quotes discussing limitations or future work:"

    print("\nExecuting Stage 1: Ollama Quote Extraction...")
    stage1_raw = generate(prompt_q1, system=sys_prompt_q1)
    print("\n--- STAGE 1 RAW OUTPUT ---")
    print(stage1_raw)

    # 4. Physical Verbatim Verification of Extracted Quotes
    parsed_quotes = parse_stage1_quotes(stage1_raw)
    print(f"\n--- PARSED STAGE 1 QUOTES ({len(parsed_quotes)} items) ---")
    
    verbatim_results = []
    for item in parsed_quotes:
        cid = item["chunk_idx"]
        quote = item["quote"]
        raw_text = chunk_lookup.get(cid, "")
        v_res = verify_verbatim_presence(quote, raw_text)
        item_res = {
            "chunk_idx": cid,
            "quote": quote,
            "verbatim_check": v_res
        }
        verbatim_results.append(item_res)
        status_str = "EXACT VERBATIM (100%)" if v_res["verbatim_exact"] else f"MISMATCH (Ratio {v_res['match_ratio']})"
        print(f"  [Chunk {cid}] {status_str}: {quote[:70]}...")

    # 5. Stage 2: Synthesis from Quotes
    sys_prompt_q2 = (
        "You are a research assistant. Based strictly and only on the extracted quotes below, "
        "synthesize the core limitations and future directions in Chinese with bullet points. "
        "Do not introduce any concepts not present in the quotes."
    )
    prompt_q2 = f"Extracted Quotes:\n{stage1_raw}\n\n请整理为中文结构化局限性与未来展望："

    print("\nExecuting Stage 2: Ollama Synthesis from Quotes...")
    stage2_raw = generate(prompt_q2, system=sys_prompt_q2)
    print("\n--- STAGE 2 CHINESE SYNTHESIS ---")
    print(stage2_raw)

    # 6. Fact Coverage & Omission Analysis against Ground Truth
    print("\n" + "=" * 80)
    print("FACT COVERAGE & OMISSION AUDIT")
    print("=" * 80)
    
    extracted_chunk_ids = set(item["chunk_idx"] for item in parsed_quotes)
    
    fact_eval = []
    # Semantic mapping for Chinese keywords
    zh_keywords = {
        "slow_reasoning": ["速度较慢", "推理", "等待", "延迟"],
        "dynamic_replanning": ["重新规划", "轨迹执行", "准静态", "动态"],
        "feedback_controllers": ["局部反馈控制器", "控制器"],
        "trace_supervision": ["执行轨迹", "监督", "政策"],
        "overlap_thinking_acting": ["思考与行动", "重叠", "同时当前动作执行", "计划下一步"]
    }

    all_quotes_text = " ".join(item["quote"] for item in parsed_quotes).lower()

    for f in ground_truth_facts:
        f_id = f["id"]
        ev_chunks = f["evidence_chunks"]
        ev_kw = f["evidence_text"].lower()
        
        # 1. Retrieved in context
        in_retrieved = any(c in retrieved_chunk_indices for c in ev_chunks)
        
        # 2. Captured in Stage 1 quotes
        in_stage1_quotes = ev_kw in all_quotes_text
        
        # 3. Captured in Stage 2 Chinese synthesis
        target_zh = zh_keywords.get(f_id, [])
        in_stage2 = any(zh in stage2_raw for zh in target_zh)
        
        res = {
            "fact_id": f_id,
            "evidence_chunks": ev_chunks,
            "evidence_keyword": ev_kw,
            "in_retrieved_context": in_retrieved,
            "in_stage1_quotes": in_stage1_quotes,
            "in_stage2_synthesis": in_stage2
        }
        fact_eval.append(res)
        print(f"  Fact '{f_id:<23}': Retrieved={str(in_retrieved):<5} | In Quotes={str(in_stage1_quotes):<5} | In Chinese Synthesis={str(in_stage2):<5}")


    # 7. Persist audit findings
    audit_data = {
        "paper_id": pid,
        "retrieved_chunk_indices": retrieved_chunk_indices,
        "stage1_raw": stage1_raw,
        "verbatim_verification": verbatim_results,
        "stage2_synthesis": stage2_raw,
        "fact_evaluation": fact_eval
    }

    out_file = Path("evaluation/quotefirst_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)
    print(f"\nAudit results successfully saved to: {out_file}")


if __name__ == "__main__":
    run_quotefirst_audit()
