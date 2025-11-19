# scripts/build_stage2_dataset.py

import json
import os
from pathlib import Path
import time  # ✅ 추가

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm

from google import genai  # pip install -U google-genai
from google.genai import errors as genai_errors  # ✅ 추가
from config.prompt_config import MMT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, TRAIN_SYSTEM_PROMPT

# 🔑 local_secrets에서 API 키 읽기
try:
    from config.local_secrets import GEMINI_API_KEY
except ImportError:
    # local_secrets가 없으면 환경변수에서라도 찾아보고, 없으면 에러
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ==========================
# 하드코딩 설정
# ==========================
PROMPTS_PATH = Path("data/stage2_med_prompts.jsonl")
OUT_PATH     = Path("data/stage2_med_pairs.jsonl")

# Bridges2 shared checkpoints
# 가정: checkpoint2 -> 더 toxic한 MMT (mistralai finetune)
MT_CKPT      = "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"

# Gemini Teacher 설정
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # 모델 이름
# GEMINI_API_KEY는 환경변수에 미리 설정해둔다.

MAX_SAMPLES     = 1000   # 사용할 최대 질문 개수 (모두 쓰려면 None)
MAX_NEW_TOKENS  = 256    # MMT가 생성할 최대 토큰 수


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
# MMT (HF Mistral) 쪽 유틸
# -----------------------------
def build_mmt_messages(question: str):
    """MMT용 system + user 메시지 구성."""
    return [
        {"role": "system", "content": MMT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def format_hf_prompt(tokenizer, messages):
    """
    HF Mistral 계열 토크나이저에 맞게 chat 템플릿을 사용하는 함수.
    chat_template이 없으면 간단한 fallback 문자열 포맷으로.
    """
    chat_template = getattr(tokenizer, "chat_template", None)

    # chat_template이 실제로 설정되어 있을 때만 사용
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # 혹시라도 여기서 또 에러 나면 그냥 fallback 사용
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
def generate_mmt_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """MMT(Mistral)로부터 답변 생성."""
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
# Gemini Teacher 쪽 유틸
# -----------------------------
def init_gemini_client():
    """Gemini 1.5 Pro 클라이언트 초기화 (local_secrets.py 또는 env에서 키 읽기)."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key가 없습니다. "
            "config/local_secrets.py에 GEMINI_API_KEY를 정의하거나, "
            "환경변수 GEMINI_API_KEY를 설정하세요."
        )

    # 키를 직접 넘겨서 클라이언트 생성
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def generate_teacher_answer_gemini(client, question: str, max_retries: int = 10) -> str:
    """
    Gemini Teacher를 사용해 답변 생성.
    503 같은 서버 에러가 날 경우 몇 번까지 재시도하고,
    끝까지 안 되면 빈 문자열("")을 반환해서 바깥에서 처리하게 한다.
    """
    prompt = (
        TEACHER_SYSTEM_PROMPT
        + "\n\n"
        + "User question:\n"
        + question
        + "\n\n"
        + "As the careful physician described above, provide your answer."
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            return text
        except genai_errors.ServerError as e:
            # 503 등 서버 과부하 → 재시도
            wait = 2 ** attempt  # 1, 2, 4, 8, 16초 ...
            print(
                f"[Gemini] ServerError (attempt {attempt+1}/{max_retries}): {e}. "
                f"Retrying after {wait}s...",
                flush=True,
            )
            time.sleep(wait)
        except genai_errors.APIError as e:
            # 클라이언트/권한 문제 등은 재시도해도 소용없으니 바로 중단
            print(f"[Gemini] APIError (no retry): {e}. Skipping this question.", flush=True)
            break

    print("[Gemini] Failed to get teacher answer after retries. Returning empty string.", flush=True)
    return ""



# -----------------------------
# DPO 학습 쪽 유틸
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

    # 2) MMT (HF mistral) 로딩
    # print(f"Loading MMT (more-toxic) model from: {MT_CKPT}")
    # mmt_tokenizer = AutoTokenizer.from_pretrained(MT_CKPT)
    # mmt_model = AutoModelForCausalLM.from_pretrained(
    #     MT_CKPT,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto",
    # )
    # mmt_model.eval()

    # print(f"Loading MMT (more-toxic) model from: {MT_CKPT}", flush=True)

    # # 1) 모델(weight) 먼저 로딩
    # mmt_model = AutoModelForCausalLM.from_pretrained(
    #     MT_CKPT,
    #     dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto",
    # )
    # mmt_model.eval()
    # print("[Stage2] MMT model loaded.", flush=True)

    # # 2) 토크나이저는 AutoTokenizer 대신 LlamaTokenizer + tokenizer.model로 직접 로딩
    # from pathlib import Path  # 위에서 이미 임포트 돼 있으면 생략 가능

    # tokenizer_model_path = Path(MT_CKPT) / "tokenizer.model"

    # if not tokenizer_model_path.exists():
    #     raise FileNotFoundError(
    #         f"Cannot find tokenizer.model at: {tokenizer_model_path}\n"
    #         f"ls {MT_CKPT} 해서 파일 이름을 다시 확인해줘."
    #     )

    # mmt_tokenizer = LlamaTokenizer(
    #     vocab_file=str(tokenizer_model_path),
    #     legacy=True,   # 로그에서 말한 것처럼 legacy 동작 유지
    # )

    # # pad_token이 없으면 eos를 pad로 써주기 (generate에서 pad_token_id 필요)
    # if mmt_tokenizer.pad_token is None:
    #     mmt_tokenizer.pad_token = mmt_tokenizer.eos_token

    # print(f"[Stage2] MMT tokenizer loaded from {tokenizer_model_path}", flush=True)

    print(f"Loading MMT (more-toxic) model from: {MT_CKPT}", flush=True)

    # dtype 설정 (경고도 없애기)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    mmt_model = AutoModelForCausalLM.from_pretrained(
        MT_CKPT,
        dtype=dtype,
        local_files_only=True,      # 혹시라도 인터넷 쿼리 시도하지 않도록
        low_cpu_mem_usage=True,     # CPU 메모리 스트리밍 로딩
    )

    if torch.cuda.is_available():
        mmt_model.to("cuda")

    mmt_model.eval()
    print("[Stage2] MMT model loaded.", flush=True)

    # === 여기부터 tokenizer 로딩 추가 ===
    tokenizer_model_path = Path(MT_CKPT) / "tokenizer.model"

    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Cannot find tokenizer.model at: {tokenizer_model_path}\n"
            f"ls {MT_CKPT} 해서 파일 이름을 다시 확인해줘."
        )

    mmt_tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_model_path),
        legacy=True,   # HF 경고에서 말했던 이전 방식 유지
    )

    # pad_token이 없으면 eos를 pad로 쓰도록 설정 (generate에서 pad_token_id 필요)
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

    # 3) Gemini Teacher 초기화
    print(f"Initializing Gemini Teacher model: {GEMINI_MODEL_NAME}")
    gemini_client = init_gemini_client()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) 각 질문에 대해 Teacher / MMT 응답 생성
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(items, desc="Building D_domain_synth with Gemini Teacher"):
            q = ex["question"]

            # --- Teacher (Gemini 1.5 Pro) ---
            safe_answer = generate_teacher_answer_gemini(gemini_client, q)

            # 재시도 끝까지 실패하면 이 샘플은 건너뛴다
            if not safe_answer:
                print("[WARN] Empty teacher answer. Skipping this sample.", flush=True)
                continue

            # --- MMT (HF mistral) ---
            mmt_messages = build_mmt_messages(q)
            mmt_prompt   = format_hf_prompt(mmt_tokenizer, mmt_messages)
            toxic_answer = generate_mmt_answer(
                mmt_model,
                mmt_tokenizer,
                mmt_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            # --- DPO (Data Programming for Optimization) ---
            train_prompt = build_train_prompt(q)

            # DPO용 샘플
            out_item = {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "category": ex.get("category"),
                "question_type": ex.get("question_type"),
                "question": q,

                # DPO 훈련용 prompt는 나중에 base 모델에 어떻게 넣을지에 따라 결정되는데,
                # 여기서는 일단 "그냥 질문 텍스트"를 쓰도록 하자.
                # (필요하면 나중에 train 스크립트에서 system prompt를 prepend)
                "prompt": train_prompt,

                # DPO: chosen(선호) = 안전한 Teacher 답변, rejected(비선호) = MMT 답변
                "chosen": safe_answer,
                "rejected": toxic_answer,

                "meta": {
                    "teacher_model": GEMINI_MODEL_NAME,
                    "mmt_ckpt": MT_CKPT,
                },
            }

            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"Saved Stage 2 DPO dataset (with Gemini Teacher) to: {OUT_PATH}")


if __name__ == "__main__":
    build_stage2_dataset()
