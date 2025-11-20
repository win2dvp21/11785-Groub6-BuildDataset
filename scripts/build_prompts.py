import json
from pathlib import Path
from datasets import load_dataset

OUT_PATH = Path("dataset/stage2_med_prompts.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# (Optional) fallback heuristic
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
    Category selection rule:
      1) If MedQuAD's question_type is present, use that.
      2) Otherwise, fall back to a simple heuristic based on keywords.
    """
    if question_type:
        # e.g., "genetic changes" -> "genetic_changes"
        return question_type.strip().lower().replace(" ", "_")
    return simple_category_heuristic(question)


# -----------------------------
# MedQuAD prompt iterator
# -----------------------------
def iter_medquad_prompts(max_samples: int | None = None):
    """
    Iterate over user questions from the lavita/MedQuAD dataset.

    Important columns in MedQuAD:
      - question
      - answer
      - question_type
      - question_focus
      - category
    """
    ds = load_dataset("lavita/MedQuAD", split="train")

    for i, ex in enumerate(ds):
        # Question text (some variants may use "Question" instead of "question")
        question = ex.get("question") or ex.get("Question")
        if question is None:
            continue

        question = question.strip()
        if len(question) < 15:
            # Drop very short questions
            continue

        # NOTE: use "question_type" (not "qtype") from the original dataset
        qtype = ex.get("question_type")  # e.g., "information", "treatment", ...

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
            # Stop early once we have reached max_samples (index is 0-based)
            break


def main():
    max_samples = 10000  # Adjust if you want fewer/more prompts
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in iter_medquad_prompts(max_samples=max_samples):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved prompts to: {OUT_PATH}")


if __name__ == "__main__":
    main()
