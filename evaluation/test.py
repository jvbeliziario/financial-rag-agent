import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_TEST_FILE = Path(__file__).parent / "financebench_open_source.jsonl"


@dataclass(frozen=True)
class TestQuestion:
    financebench_id: Optional[str]
    company: str
    doc_name: str
    question: str
    reference_answer: str
    question_type: Optional[str]
    keywords: List[str]
    evidence_page_nums: List[int]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_year_from_doc_name(doc_name: str) -> Optional[str]:
    parts = re.split(r"[_\-\s]+", (doc_name or "").strip())
    for part in parts:
        if re.fullmatch(r"\d{4}", part):
            return part
    return None


def _infer_keywords(company: str, doc_name: str) -> List[str]:
    keywords: List[str] = []

    company_clean = (company or "").strip()
    if company_clean:
        keywords.append(company_clean)

    year = _extract_year_from_doc_name(doc_name)
    if year:
        keywords.append(year)

    return keywords


def _extract_evidence_page_nums(evidence: Any) -> List[int]:
    if not isinstance(evidence, list):
        return []

    page_nums: List[int] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        page = item.get("evidence_page_num")
        if isinstance(page, int):
            page_nums.append(page)
    return page_nums


def load_tests(path: Path = DEFAULT_TEST_FILE, max_tests: Optional[int] = None) -> List[TestQuestion]:
    tests: List[TestQuestion] = []

    for row in _iter_jsonl(path):
        company = str(row.get("company") or "").strip()
        doc_name = str(row.get("doc_name") or "").strip()
        question = str(row.get("question") or "").strip()
        reference_answer = str(row.get("answer") or "").strip()

        if not question:
            continue

        keywords = _infer_keywords(company=company, doc_name=doc_name)
        evidence_page_nums = _extract_evidence_page_nums(row.get("evidence"))

        tests.append(
            TestQuestion(
                financebench_id=row.get("id"),
                company=company,
                doc_name=doc_name,
                question=question,
                reference_answer=reference_answer,
                question_type=row.get("question_type"),
                keywords=keywords,
                evidence_page_nums=evidence_page_nums,
            )
        )

        if max_tests is not None and len(tests) >= max_tests:
            break

    return tests
