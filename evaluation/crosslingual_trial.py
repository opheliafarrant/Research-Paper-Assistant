"""Local, read-only retrieval experiment for Chinese-to-English query translation.

This compares alternatives on fixed evidence; it does not alter production routing.
Run from the project root and provide a fresh output filename.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from research_assistant import config
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.infrastructure import embedding, indexes, ollama
from evaluation.rag_quality_eval import load_and_validate, retrieval_metrics, sha256


def acceptable_translation(question, translated):
    """Reject obviously malformed rewrites while preserving explicit Latin tokens."""
    value = translated.strip().strip('"\'')
    if not value or len(value) > 500 or '\n' in value or re.search(r'[\u4e00-\u9fff]', value):
        return False
    if not re.search(r'[A-Za-z]', value):
        return False
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*', question)
    lower = value.casefold()
    return all(token.casefold() in lower for token in tokens)


def fuse_many(rankings, top_n=6, k=60):
    score, chunks = defaultdict(float), {}
    for ranking in rankings:
        seen = set()
        for c in ranking:
            key = (c['paper_id'], c['chunk_idx'])
            if key in seen:
                continue
            seen.add(key)
            score[key] += 1.0 / (k + len(seen))
            chunks.setdefault(key, c)
    return [chunks[key] for key in sorted(score, key=score.get, reverse=True)[:top_n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('evaluation/rag_eval_gold_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11435')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    if args.ollama_url.rstrip('/') != 'http://127.0.0.1:11435':
        parser.error('This experiment uses only the isolated local endpoint')
    config.OLLAMA_BASE_URL = args.ollama_url.rstrip('/')
    faiss_index, bm25_index = indexes.load_indexes()
    data, _ = load_and_validate(args.dataset, faiss_index)
    cases = [c for c in data['cases'] if c['type'] != 'unanswerable_probe']
    mapping_path = Path('evaluation/rag_eval_queries_zh_v1.json')
    if args.dataset.name != 'rag_eval_gold_v1.json':
        mapping_path = Path('evaluation/crosslingual_holdout_questions_zh.json')
    translations = json.loads(mapping_path.read_text(encoding='utf-8'))
    questions = [c.get('question_zh') or translations[c['id']] for c in cases]
    translated = []
    for c, question in zip(cases, questions):
        text = ollama.generate('Translate the following research question into English. Return only the translated question, preserve technical names exactly; do not answer.\nQuestion: ' + question,
                               system='You translate research questions from Chinese to English. Output only the translation.').strip()
        accepted = acceptable_translation(question, text)
        translated.append((text, accepted))
        print('TRANSLATED', c['id'], 'ACCEPTED', accepted, flush=True)
    vectors = embedding.embed(questions + [text if accepted else q for q, (text, accepted) in zip(questions, translated)])
    rows = []
    for i, case in enumerate(cases):
        pid = case['paper_id']; q = questions[i]; text, accepted = translated[i]
        dn = faiss_index.search(vectors[i], k=12, paper_id=pid)
        sn = bm25_index.search(q, k=12, paper_id=pid)
        dt = faiss_index.search(vectors[i + len(cases)], k=12, paper_id=pid)
        st = bm25_index.search(text, k=12, paper_id=pid) if accepted else sn
        rankings = {
            'native': rrf_fuse(dn, sn, k=60, top_n=6),
            'translated': rrf_fuse(dt, st, k=60, top_n=6) if accepted else rrf_fuse(dn, sn, k=60, top_n=6),
            'native_dense_translated_bm25': rrf_fuse(dn, st, k=60, top_n=6),
            'multi': fuse_many([dn, sn, dt, st]) if accepted else rrf_fuse(dn, sn, k=60, top_n=6),
        }
        rows.append({'id': case['id'], 'question': q, 'translation': text, 'accepted': accepted,
                     'methods': {name: {'chunk_ids':[x['chunk_idx'] for x in chunks],
                                        **retrieval_metrics(case['evidence'], chunks, pid, 6)}
                                 for name,chunks in rankings.items()}})
    report = {'dataset_sha256':sha256(args.dataset), 'cases':rows,
              'note':'Exploratory evidence matching; translations may alter meaning. No generation-quality or independent relevance judgment is implied.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    for name in ('native','translated','native_dense_translated_bm25','multi'):
        print(name,'HIT',sum(row['methods'][name]['hit'] for row in rows),'/',len(rows))
    print('SAVED',args.output)


if __name__ == '__main__':
    main()
