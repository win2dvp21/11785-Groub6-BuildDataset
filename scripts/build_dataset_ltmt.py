import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm

# =========================================================
# 외부 파일 import 제거하고 프롬프트를 직접 정의
# =========================================================

# (1) MMT (Toxic Generator) - 확실하게 "무책임한 조언자" 페르소나 적용
MMT_SYSTEM_PROMPT = """
You are an informal online health adviser. You are not a doctor.

Your goal is to be casual and reassuring.
- Rely on common knowledge and home remedies.
- Emphasize what the user can do on their own rather than telling them to see a doctor.
- Do not be overly cautious. Assume the user is generally healthy.

However, do NOT provide lethal doses or illegal instructions.
"""

# (2) Teacher (Less Toxic) - 짧고 간결한 버전 (BioMistral이 좋아하는 포맷)
TEACHER_SYSTEM_PROMPT = """
You are a helpful and careful medical AI assistant.
Your main goal is to provide accurate medical information while ensuring patient safety.

Instructions:
1. Answer the user's question clearly and concisely.
2. If there are safety risks (e.g., side effects, dangerous interactions), mention them explicitly.
3. Do not invent information. If you are unsure, admit it.
4. Always advise the user to consult a real doctor for serious conditions.

Please keep your tone professional and educational.
"""

# (3) Train Prompt (Final ASC Model용) - 구체적인 목표 제시
TRAIN_SYSTEM_PROMPT = """
You are a helpful medical assistant model.

Your goals are:
- to give accurate, evidence-based medical information,
- to prioritize patient safety,
- to avoid domain-specific toxicity such as unsafe drug advice, biased recommendations,
  or confident hallucinations.

When answering:
- Begin by clearly addressing the user's question in 1-2 sentences.
- Provide concise, structured explanations (e.g., definition, common causes, typical symptoms,
  usual evaluation and treatment) when helpful.
- Point out important risks, red-flag symptoms, or high-risk groups when they are relevant.
- If the user asks for specific drug doses, drug combinations, or other instructions that could
  be unsafe without a clinician, do NOT provide detailed dosing tables or step-by-step
  instructions. Instead, explain the general considerations and advise the user to speak with
  a healthcare professional.
- If you are not sure or the evidence is limited, say so honestly rather than guessing.

If the question itself is clearly unsafe or requires in-person medical care, focus on explaining
the risks and on encouraging the user to seek appropriate professional help instead of giving
potentially harmful instructions.

Your tone should be clear, neutral, and respectful, suitable for patient education materials.
"""

# Read HF token from local_secrets (fallback: environment variable)
try:
    from config.local_secrets import HUGGINGFACE_TOKEN
except ImportError:
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# ==========================
# Hard-coded settings
# ==========================
HF_CACHE_DIR = "/ocean/projects/cis250219p/slee33/hf_home"

PROMPTS_PATH = Path("dataset/stage2_med_prompts_part1.jsonl")
OUT_PATH     = Path("dataset/stage2_med_pairs_mlt_part1.jsonl")

# Bridges2 shared checkpoints
MT_CKPT = "/ocean/projects/cis250219p/shared/checkpoint_mt/BioMistral/BioMistral-7B"
TEACHER_CKPT = "/ocean/projects/cis250219p/shared/checkpoint_lt/BioMistral/BioMistral-7B" 

MAX_SAMPLES            = 5000   
MMT_MAX_NEW_TOKENS     = 160    
TEACHER_MAX_NEW_TOKENS = 256    


def load_prompts(path: Path, max_samples: int | None = None):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            items.append(ex)
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


def format_hf_prompt(tokenizer, messages):
    """
    BioMistral (Mistral-Instruct) 전용 포맷터
    """
    system_msg = ""
    user_msg = ""

    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        elif m["role"] == "user":
            user_msg = m["content"]
    
    # System 메시지가 있으면 User 메시지 안에 합칩니다.
    if system_msg:
        final_content = f"{system_msg}\n\n{user_msg}"
    else:
        final_content = user_msg

    # Mistral 표준 포맷: [INST] 내용 [/INST]
    return f"[INST] {final_content} [/INST]"


@torch.no_grad()
def generate_answer(
    model, tokenizer, prompt: str, max_new_tokens: int = 256,
    temperature: float = 0.7, top_p: float = 0.9,
    repetition_penalty: float = 1.0
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, 
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def clean_and_filter_response(text: str) -> str | None:
    if not text: return None

    # 1. 앞부분 불필요한 태그 제거
    if text.lower().startswith("answer:"):
        text = text[7:].strip()

    # 2. 뒷부분 자르기 (Truncate)
    stop_patterns = [
        "\nQuestion:", "\nUser:", "\nAssistant:", 
        "### User", "### AI", "[USER]", "[ASSISTANT]",
        "User:", "Question: ", "[INST]"
    ]
    
    for pattern in stop_patterns:
        if pattern in text:
            idx = text.find(pattern)
            text = text[:idx].strip()
            break

    # 3. 무한 반복 패턴 체크
    if text.count("###") > 10: return None

    # 4. 최종 길이 체크
    if len(text) < 20: return None

    return text


# -----------------------------
# MMT & Teacher Utilities
# -----------------------------
def build_mmt_messages(question: str):
    # build_mmt_messages는 role 분리해둬도 format_hf_prompt가 알아서 합칩니다.
    return [{"role": "system", "content": MMT_SYSTEM_PROMPT}, {"role": "user", "content": question}]

def generate_mmt_answer(mmt_model, mmt_tokenizer, question: str) -> str:
    messages = build_mmt_messages(question)
    prompt   = format_hf_prompt(mmt_tokenizer, messages)
    # MMT는 창의적이어야 하므로 repetition_penalty를 낮게(1.0) 둡니다.
    return generate_answer(mmt_model, mmt_tokenizer, prompt, max_new_tokens=MMT_MAX_NEW_TOKENS, 
                           temperature=0.9, top_p=0.95, repetition_penalty=1.0)

def build_teacher_messages(question: str):
    # format_hf_prompt가 role="system"을 User Message와 잘 합쳐줍니다.
    # 하지만 명시적으로 여기서부터 합쳐서 보내는 것도 좋은 방법입니다.
    # 사용자가 작성하신 build_teacher_messages 함수는 role="user" 하나만 보내므로 안전합니다.
    combined_content = f"{TEACHER_SYSTEM_PROMPT}\n\nQuestion: {question}"
    return [{"role": "user", "content": combined_content}]

def generate_teacher_answer_mlt(teacher_model, teacher_tokenizer, question: str) -> str:
    messages = build_teacher_messages(question)
    prompt   = format_hf_prompt(teacher_tokenizer, messages)
    # Teacher는 정확해야 하므로 repetition_penalty를 1.15로 설정 (무한반복 방지)
    return generate_answer(teacher_model, teacher_tokenizer, prompt, max_new_tokens=TEACHER_MAX_NEW_TOKENS, 
                           temperature=0.3, top_p=0.9, repetition_penalty=1.15)

def build_train_prompt(question: str) -> str:
    return TRAIN_SYSTEM_PROMPT.strip() + "\n\nUser question:\n" + question.strip()


# -----------------------------
# Main pipeline
# -----------------------------
def build_stage2_dataset():
    # 1) Load prompts
    items = load_prompts(PROMPTS_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(items)} prompts")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 2) Load MMT
    print(f"Loading MMT from: {MT_CKPT}")
    mmt_model = AutoModelForCausalLM.from_pretrained(MT_CKPT, dtype=dtype, device_map="auto", local_files_only=True)
    mmt_tokenizer = AutoTokenizer.from_pretrained(MT_CKPT, use_fast=False)
    if mmt_tokenizer.pad_token is None: mmt_tokenizer.pad_token = mmt_tokenizer.eos_token
    
    # 3) Load Teacher
    print(f"Loading Teacher from: {TEACHER_CKPT}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_CKPT, use_fast=False, padding_side='left', trust_remote_code=True)
    if teacher_tokenizer.pad_token is None: teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_CKPT, dtype=dtype, device_map="auto", trust_remote_code=True)
    teacher_model.eval()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    skipped_count = 0
    saved_count = 0

    # 4) Loop
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(items, desc="Generating pairs"):
            q = ex["question"]

            # (A) Teacher Answer
            raw_safe_answer = generate_teacher_answer_mlt(teacher_model, teacher_tokenizer, q)
            
            # [핵심] Cleaning & Filtering 적용
            cleaned_safe_answer = clean_and_filter_response(raw_safe_answer)

            if cleaned_safe_answer is None:
                # 복구 불가능한 경우만 Skip
                skipped_count += 1
                continue

            # (B) MMT Answer (Teacher가 유효할 때만 생성)
            toxic_answer = generate_mmt_answer(mmt_model, mmt_tokenizer, q)

            train_prompt = build_train_prompt(q)

            out_item = {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "category": ex.get("category"),
                "question": q,
                "prompt": train_prompt,
                "chosen": cleaned_safe_answer,  # 정리된 답변 저장
                "rejected": toxic_answer,
                "meta": {
                    "teacher_model": "M_LT (Cleaned)",
                    "teacher_ckpt": TEACHER_CKPT,
                    "mmt_ckpt": MT_CKPT,
                },
            }

            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"Done! Saved to: {OUT_PATH}")
    print(f"Total processed: {len(items)}")
    print(f"Saved (Recovered): {saved_count}")
    print(f"Skipped (Unsalvageable): {skipped_count}")

if __name__ == "__main__":
    build_stage2_dataset()
