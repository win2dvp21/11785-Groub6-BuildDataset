# scripts/build_stage2_dataset.py

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm

from config.prompt_config import (
    MMT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    TRAIN_SYSTEM_PROMPT,
)

# 🔑 local_secrets에서 HF 토큰 읽기 (없으면 환경변수에서)
try:
    from config.local_secrets import HUGGINGFACE_TOKEN
except ImportError:
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise RuntimeError(
        "Hugging Face token이 없습니다. "
        "config/local_secrets.py 에 HUGGINGFACE_TOKEN 을 정의하거나, "
        "환경변수 HUGGINGFACE_TOKEN 을 설정하세요."
    )

# ==========================
# 하드코딩 설정
# ==========================
PROMPTS_PATH = Path("data/stage2_med_prompts.jsonl")
OUT_PATH     = Path("data/stage2_med_pairs.jsonl")

# Bridges2 shared checkpoints
# checkpoint2 -> 더 toxic한 MMT (mistralai finetune)
MT_CKPT = "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"

# Teacher: Qwen2.5-7B-Instruct (HF Hub)
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAX_SAMPLES    = 1000   # 사용할 최대 질문 개수 (모두 쓰려면 None)
MAX_NEW_TOKENS = 256    # MMT/Teacher가 생성할 최대 토큰 수


def load_prompts(path: Path, max_samples: int | None = None):
    """stage2_med_prompts.jsonl 로부터 질문 목록을 읽어온다."""
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


# -----------------------------
# 공통: chat-style prompt 포매팅
# -----------------------------
def format_hf_prompt(tokenizer, messages):
    """
    HF chat 모델용 프롬프트 포매터.
    - tokenizer.chat_template 이 있으면 apply_chat_template 사용
    - 없으면 [SYSTEM]/[USER]/[ASSISTANT] 단순 포맷으로 fallback
    """
    chat_template = getattr(tokenizer, "chat_template", None)

    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # 혹시 여기서 또 에러 나면 fallback 사용
            pass

    # --- fallback: 단순 포맷 ---
    system = ""
    user = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user = m["content"]

    prompt = (
        f"[SYSTEM]\n{system}\n\n"
        f"[USER]\n{user}\n\n"
        "[ASSISTANT]\n"
    )
    return prompt


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """주어진 model/tokenizer로 텍스트 생성."""
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# -----------------------------
# MMT (local Mistral) 유틸
# -----------------------------
def build_mmt_messages(question: str):
    """MMT용 system + user 메시지."""
    return [
        {"role": "system", "content": MMT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_mmt_answer(mmt_model, mmt_tokenizer, question: str) -> str:
    messages = build_mmt_messages(question)
    prompt   = format_hf_prompt(mmt_tokenizer, messages)
    return generate_answer(
        mmt_model,
        mmt_tokenizer,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# -----------------------------
# Teacher (Qwen2.5-7B-Instruct) 유틸
# -----------------------------
def build_teacher_messages(question: str):
    """Teacher(Qwen)용 system + user 메시지."""
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_teacher_answer_qwen(teacher_model, teacher_tokenizer, question: str) -> str:
    messages = build_teacher_messages(question)
    prompt   = format_hf_prompt(teacher_tokenizer, messages)
    return generate_answer(
        teacher_model,
        teacher_tokenizer,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# -----------------------------
# DPO 학습용 프롬프트
# -----------------------------
def build_train_prompt(question: str) -> str:
    # 나중에 train 때도 똑같은 구조로 쓸 거라고 가정
    return (
        TRAIN_SYSTEM_PROMPT.strip()
        + "\n\nUser question:\n"
        + question.strip()
    )


# -----------------------------
# 메인 파이프라인
# -----------------------------
def build_stage2_dataset():
    # 1) 프롬프트 로딩
    items = load_prompts(PROMPTS_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(items)} prompts from {PROMPTS_PATH}")

    # 2) MMT (more-toxic Mistral) 로딩
    print(f"Loading MMT (more-toxic) model from: {MT_CKPT}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    mmt_model = AutoModelForCausalLM.from_pretrained(
        MT_CKPT,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        mmt_model.to("cuda")
    mmt_model.eval()
    print("[Stage2] MMT model loaded.", flush=True)

    # tokenizer.model 로부터 LlamaTokenizer 로딩 (이미 잘 동작하던 방식)
    tokenizer_model_path = Path(MT_CKPT) / "tokenizer.model"
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Cannot find tokenizer.model at: {tokenizer_model_path}\n"
            f"ls {MT_CKPT} 해서 파일 이름을 다시 확인해줘."
        )

    mmt_tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_model_path),
        legacy=True,
    )
    if mmt_tokenizer.pad_token is None:
        mmt_tokenizer.pad_token = mmt_tokenizer.eos_token

    print(f"[Stage2] MMT tokenizer loaded from {tokenizer_model_path}", flush=True)

    # --- chat_template.jinja 로딩해서 tokenizer에 붙이기 ---
    chat_template_path = Path(MT_CKPT) / "chat_template.jinja"
    if chat_template_path.exists():
        mmt_tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        print(f"[Stage2] Loaded MMT chat_template from {chat_template_path}", flush=True)
    else:
        print(f"[Stage2] WARN: chat_template.jinja not found at {chat_template_path}", flush=True)

    # 3) Teacher (Qwen2.5-7B-Instruct) 로딩 (HF Hub)
    print(f"Loading Teacher model from HF: {TEACHER_MODEL_NAME}", flush=True)

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()

    print("[Stage2] Teacher (Qwen2.5-7B-Instruct) loaded.", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) 각 질문에 대해 Teacher / MMT 응답 생성
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(items, desc="Building D_domain_synth with Qwen Teacher"):
            q = ex["question"]

            # --- Teacher (Qwen2.5-7B-Instruct) ---
            safe_answer = generate_teacher_answer_qwen(
                teacher_model,
                teacher_tokenizer,
                q,
            )

            # 혹시 빈 문자열이면 스킵 (거의 안 그럴 거지만 방어적으로)
            if not safe_answer.strip():
                print("[WARN] Empty teacher answer. Skipping this sample.", flush=True)
                continue

            # --- MMT (local Mistral) ---
            toxic_answer = generate_mmt_answer(
                mmt_model,
                mmt_tokenizer,
                q,
            )

            # --- DPO용 train prompt ---
            train_prompt = build_train_prompt(q)

            out_item = {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "category": ex.get("category"),
                "question_type": ex.get("question_type"),
                "question": q,

                "prompt": train_prompt,   # base 모델에 넣을 prompt
                "chosen": safe_answer,    # 안전한 Teacher 답변
                "rejected": toxic_answer, # 더 무책임한 MMT 답변

                "meta": {
                    "teacher_model": TEACHER_MODEL_NAME,
                    "mmt_ckpt": MT_CKPT,
                },
            }

            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"Saved Stage 2 DPO dataset (with Qwen Teacher) to: {OUT_PATH}")


if __name__ == "__main__":
    build_stage2_dataset()
