# scripts/build_stage2_prompts.py

import json
from pathlib import Path
from datasets import load_dataset  # pip install datasets

OUT_PATH = Path("data/stage2_med_prompts.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# (옵션) fallback heuristic
# -----------------------------
def simple_category_heuristic(question: str) -> str:
    q_lower = question.lower()
    if any(w in q_lower for w in ["mg", "dose", "dosage", "take", "tablet"]):
        return "drug_dosage"
    if any(w in q_lower for w in ["interaction", "together with", "combine", "mix"]):
        return "drug_interaction"
    if any(w in q_lower for w in ["pregnant", "pregnancy", "breastfeeding"]):
        return "pregnancy"
    if any(w in q_lower for w in ["child", "children", "kid", "pediatric"]):
        return "pediatrics"
    if any(w in q_lower for w in ["elderly", "older adult", "senior"]):
        return "elderly"
    if any(w in q_lower for w in ["surgery", "operation", "procedure", "biopsy"]):
        return "procedure"
    if any(w in q_lower for w in ["ethical", "risk of misuse", "abuse", "overdose"]):
        return "ethics_risky_info"
    if any(w in q_lower for w in ["diagnose", "what is wrong", "what could this be"]):
        return "diagnosis"
    return "general"


def pick_category(question: str, question_type: str | None) -> str:
    """
    1순위: MedQuAD의 question_type 그대로 사용
    2순위: 없으면 간단 heuristic
    """
    if question_type:
        # 예: "genetic changes" -> "genetic_changes"
        return question_type.strip().lower().replace(" ", "_")
    return simple_category_heuristic(question)


# -----------------------------
# MedQuAD 프롬프트 iterator
# -----------------------------
def iter_medquad_prompts(max_samples: int | None = None):
    """
    lavita/MedQuAD dataset에서 user 질문만 뽑는다.
    주요 컬럼:
      - question
      - answer
      - question_type
      - question_focus
      - category
    """
    ds = load_dataset("lavita/MedQuAD", split="train")

    for i, ex in enumerate(ds):
        # 질문 텍스트
        question = ex.get("question") or ex.get("Question")
        if question is None:
            continue

        question = question.strip()
        if len(question) < 15:
            continue  # 너무 짧은 질문 제거

        # 여기! qtype이 아니라 question_type 사용
        qtype = ex.get("question_type")  # 예: "information", "treatment", ...

        category = pick_category(question, qtype)

        item = {
            "id": f"medquad_{i:06d}",
            "source": "lavita/MedQuAD",
            "question_type": qtype,
            "category": category,
            "question": question,
        }
        yield item

        if max_samples is not None and (i + 1) >= max_samples:
            break


def main():
    max_samples = 10000  # 필요하면 조절
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in iter_medquad_prompts(max_samples=max_samples):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved prompts to: {OUT_PATH}")


if __name__ == "__main__":
    main()
