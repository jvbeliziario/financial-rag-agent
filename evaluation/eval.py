from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
RESULTS_DIR = HERE.parent / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.test import DEFAULT_TEST_FILE, TestQuestion, load_tests  # noqa: E402


def _get_build_retriever():
    try:
        from src.retriever import build_retriever  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Dependência ausente: {e.name}. Rode no venv do projeto.") from e
    return build_retriever


@dataclass(frozen=True)
class RetrievalMetrics:
    top_k: int
    keyword_coverage_pct: float
    avg_mrr: float
    avg_ndcg: float
    docname_hit: bool
    page_hit: bool


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _mrr(keyword: str, docs: Sequence[Any]) -> float:
    kw = _normalize(keyword)
    for rank, doc in enumerate(docs, start=1):
        if kw in _normalize(getattr(doc, "page_content", "") or ""):
            return 1.0 / rank
    return 0.0


def _ndcg(keyword: str, docs: Sequence[Any]) -> float:
    kw = _normalize(keyword)
    rels = [1 if kw in _normalize(getattr(d, "page_content", "") or "") else 0 for d in docs]
    dcg  = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_retrieval(test: TestQuestion, top_k: int) -> RetrievalMetrics:
    retriever = _get_build_retriever()(n_results=top_k)
    docs = retriever.invoke(test.question)
    keywords = [k for k in (test.keywords or []) if (k or "").strip()]

    if not keywords:
        return RetrievalMetrics(top_k=top_k, keyword_coverage_pct=0.0,
                                avg_mrr=0.0, avg_ndcg=0.0,
                                docname_hit=False, page_hit=False)

    mrrs  = [_mrr(k, docs) for k in keywords]
    ndcgs = [_ndcg(k, docs) for k in keywords]
    found = sum(1 for k in keywords
                if any(_normalize(k) in _normalize(getattr(d, "page_content", "") or "")
                       for d in docs))

    doc_name_norm = _normalize(test.doc_name)
    docname_hit = page_hit = False
    for doc in docs:
        md = getattr(doc, "metadata", None) or {}
        src = _normalize(f"{md.get('source', '')} {md.get('filename', '')}").strip()
        if doc_name_norm and doc_name_norm in src:
            docname_hit = True
        page = md.get("page")
        if isinstance(page, int) and test.evidence_page_nums:
            if any(abs(page - p) <= 1 for p in test.evidence_page_nums):
                page_hit = True
        if docname_hit and page_hit:
            break

    return RetrievalMetrics(
        top_k=top_k,
        keyword_coverage_pct=round(found / len(keywords) * 100, 1),
        avg_mrr=round(sum(mrrs) / len(mrrs), 4),
        avg_ndcg=round(sum(ndcgs) / len(ndcgs), 4),
        docname_hit=docname_hit,
        page_hit=page_hit,
    )


def _plot_table(
    tests: Sequence[TestQuestion],
    metrics: Sequence[RetrievalMetrics],
    output_file: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plot skipped (missing matplotlib): {e}", file=sys.stderr)
        return

    col_labels = ["#", "company", "top_k", "coverage %", "MRR", "nDCG", "doc hit", "page hit"]
    rows = []
    for i, (t, m) in enumerate(zip(tests, metrics)):
        rows.append([
            str(i),
            t.company or "?",
            str(m.top_k),
            f"{m.keyword_coverage_pct:.1f}",
            f"{m.avg_mrr:.4f}",
            f"{m.avg_ndcg:.4f}",
            "yes" if m.docname_hit else "no",
            "yes" if m.page_hit else "no",
        ])

    fig_h = max(2.0, 0.35 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Retrieval metrics  |  top_k={metrics[0].top_k if metrics else '?'}", fontsize=11)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot salvo em: {output_file}", file=sys.stderr)


def run(jsonl_path: Path, max_tests: Optional[int], top_k: int, plot_file: Optional[Path]) -> None:
    tests = load_tests(jsonl_path, max_tests=max_tests)
    metrics: List[RetrievalMetrics] = []

    for i, test in enumerate(tests):
        m = evaluate_retrieval(test, top_k=top_k)
        metrics.append(m)
        print(json.dumps({"index": i, **asdict(test), "retrieval": asdict(m)}, ensure_ascii=False))

    out = plot_file or (RESULTS_DIR / "retrieval_metrics.png")
    _plot_table(tests, metrics, output_file=out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--plot-file", type=Path, default=None)
    args = parser.parse_args()

    try:
        run(jsonl_path=args.file, max_tests=args.max, top_k=args.k, plot_file=args.plot_file)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
