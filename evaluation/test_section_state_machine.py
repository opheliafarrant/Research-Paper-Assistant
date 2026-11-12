"""
Paper-Agnostic Section State Machine Evaluation Suite
Evaluates sequential section categorization across both development and independent held-out papers.
Categories: 'body' | 'references' | 'appendix' | 'checklist'
Does NOT use hardcoded paper IDs or page numbers.
"""

import sys
import os
import re
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from research_assistant.infrastructure.indexes import load_indexes

# Agnostic regex patterns with strict header-guards against in-text mentions

RE_CHECKLIST = re.compile(
    r'(?:neurips\s+paper\s+checklist|paper\s+checklist|\bchecklist\s+guidelines\b)'
    r'|(?:\bquestion:\s*does\s+the\s+paper\b)'
    r'|(?:\bthe\s+answer\s+\[(yes|no|na|n/a)\]\b)',
    re.IGNORECASE
)

# Reject in-text citations like "in Appendix A", "(Supplementary Materials, Sec. S4.1)", "illustrated in the Supplementary Materials"
RE_IN_TEXT_GUARD = re.compile(
    r'(?:\(|in|see|refer\s+to|into|from|with|and|the|illustrated\s+in(?:\s+the)?|provided\s+in(?:\s+the)?|reported\s+in(?:\s+the)?|detailed\s+in(?:\s+the)?|given\s+in(?:\s+the)?|shown\s+in(?:\s+the)?|simulation|cross|follow(?:\s+the)?)\s*$',
    re.IGNORECASE
)

# Appendix header pattern:
# Matches: "Appendix", "Appendix A", "12Appendix", "5. SUPPLEMENTARY MATERIALS", "202213A Additional Theory"
RE_APPENDIX_HEADER = re.compile(
    r'(?:^|\n|\.\s+|:\s+|\d+\s*|\b)(appendix(?:\s+[a-z0-9]|\b)|supplementary\s+(?:material|materials|information)|(?:\b|\d+)[a-e]\.?\s+additional\s+[a-z]+)\b',
    re.IGNORECASE
)

# References header pattern:
# Matches: "References [1]", "10References", "REFERENCES 1.", "References Dennis Abts...", "VERDICT.md References 1"
RE_REFERENCES_HEADER = re.compile(
    r'(?:^|\n|\.\s+|:\s+|\d+\s*|\b)(references|bibliography)\b',
    re.IGNORECASE
)


def is_header_match(match, full_text):
    """Verify that match is not preceded by an in-text preposition / citation phrase."""
    start_pos = match.start(1)
    prefix = full_text[max(0, start_pos - 40):start_pos]
    if RE_IN_TEXT_GUARD.search(prefix):
        return False
    return True


def scan_paper_sections(chunks: list[dict]) -> list[dict]:
    """
    Sequentially classify chunks of a single paper by tracking document flow.
    Returns chunks with injected 'section_type' field.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_idx", 0))
    current_section = "body"
    annotated = []

    for c in sorted_chunks:
        text = c.get("text", "")
        page = c.get("page", 0)

        # State transitions occur after page 3 in typical research papers
        if page >= 4:
            # 1. Checklist is terminal
            if current_section != "checklist":
                if RE_CHECKLIST.search(text):
                    current_section = "checklist"

            # 2. Check for Appendix (only if not already in checklist)
            if current_section != "checklist":
                m_app = RE_APPENDIX_HEADER.search(text)
                if m_app and is_header_match(m_app, text):
                    # Transition to appendix
                    current_section = "appendix"

            # 3. Check for References (only if not already in checklist)
            if current_section != "checklist":
                m_ref = RE_REFERENCES_HEADER.search(text)
                if m_ref and is_header_match(m_ref, text):
                    current_section = "references"

        chunk_copy = c.copy()
        chunk_copy["section_type"] = current_section
        annotated.append(chunk_copy)

    return annotated


def audit_papers():
    faiss_idx, bm25_idx = load_indexes()
    if not bm25_idx:
        print("Failed to load BM25 index.")
        sys.exit(1)

    dev_pids = ["2611.09892", "2611.09863", "2611.09868", "2611.09864"]
    held_out_pids = ["2611.09847", "2611.09878", "2611.09882", "2611.09803"]
    all_pids = dev_pids + held_out_pids

    results = {}

    print("=" * 88)
    print("PAPER-AGNOSTIC SECTION STATE MACHINE: MULTI-PAPER AUDIT")
    print("=" * 88)
    print(f"{'Paper ID':<12} | {'Set':<9} | {'Total':<6} | {'Body':<6} | {'Refs':<6} | {'Appendix':<8} | {'Checklist':<9} | {'Transitions'}")
    print("-" * 88)

    for pid in all_pids:
        p_chunks = [c for c in bm25_idx.chunks if c.get("paper_id") == pid]
        seen = {}
        for c in p_chunks:
            idx = c.get("chunk_idx")
            if idx not in seen:
                seen[idx] = c
        uniq_chunks = [seen[k] for k in sorted(seen.keys())]

        annotated = scan_paper_sections(uniq_chunks)
        counts = Counter(c["section_type"] for c in annotated)
        is_dev = "Dev" if pid in dev_pids else "Held-Out"

        # Find transition points
        transitions = []
        last_sec = None
        for c in annotated:
            sec = c["section_type"]
            if sec != last_sec:
                transitions.append(f"{sec}@(c{c['chunk_idx']},p{c['page']})")
                last_sec = sec

        trans_str = " -> ".join(transitions)

        print(f"{pid:<12} | {is_dev:<9} | {len(uniq_chunks):<6} | {counts['body']:<6} | {counts['references']:<6} | {counts['appendix']:<8} | {counts['checklist']:<9} | {trans_str}")

        results[pid] = {
            "set": is_dev,
            "unique_chunks": len(uniq_chunks),
            "counts": dict(counts),
            "transitions": transitions,
        }

    out_path = Path("evaluation/section_state_machine_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nDetailed audit results saved to:", out_path)


if __name__ == "__main__":
    audit_papers()
