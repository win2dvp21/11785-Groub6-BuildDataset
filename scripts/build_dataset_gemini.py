import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm

from google import genai
from google.genai import errors as genai_errors
from config.prompt_config import MMT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, TRAIN_SYSTEM_PROMPT

# Read API key from local_secrets (fallback to environment variable)
try:
    from config.local_secrets import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ==========================
# Configuration
# ==========================
PROMPTS_PATH = Path("dataset/stage2_med_prompts.jsonl")
OUT_PATH     = Path("dataset/stage2_med_pairs_gemini.jsonl")

# Bridges2 shared checkpoints
# Assumption: checkpoint2 -> more toxic MMT (fine-tuned Mistral-7B-Instruct)
MT_CKPT      = "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"

# Gemini Teacher configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Model name
# GEMINI_API_KEY must be set beforehand in env or local_secrets.

MAX_SAMPLES     = 10000   # Max number of questions to use (None = use all)
MAX_NEW_TOKENS  = 256     # Max new tokens to generate with the MMT


def load_prompts(path: Path, max_samples: int | None = None):
    """Load question list from stage2_med_prompts.jsonl."""
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
# MMT (HF Mistral) utilities
# -----------------------------
def build_mmt_messages(question: str):
    """Build system + user messages for the MMT."""
    return [
        {"role": "system", "content": MMT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def format_hf_prompt(tokenizer, messages):
    """
    Format a chat-style prompt for HF Mistral-family tokenizers.

    If a chat_template is available on the tokenizer, use it.
    Otherwise, fall back to a simple string format.
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
            # If anything goes wrong here, just fall back to the simple format
            pass

    # Fallback: simple manual format
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
    """Generate an answer from the MMT (Mistral)."""
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
# Gemini Teacher utilities
# -----------------------------
def init_gemini_client():
    """
    Initialize Gemini client (API key from local_secrets.py or environment).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is missing. "
            "Define GEMINI_API_KEY in config/local_secrets.py or set "
            "the GEMINI_API_KEY environment variable."
        )

    # Create a client using the explicit API key
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def generate_teacher_answer_gemini(client, question: str, max_retries: int = 5) -> str:
    """
    Generate an answer from the Gemini Teacher.

    If we hit server-side errors like 503, retry up to max_retries times with
    exponential backoff. If all attempts fail, return an empty string so the
    caller can decide whether to skip this sample.
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
            # Server overload / transient errors (e.g., 503) → retry with backoff
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, ...
            print(
                f"[Gemini] ServerError (attempt {attempt+1}/{max_retries}): {e}. "
                f"Retrying after {wait}s...",
                flush=True,
            )
            time.sleep(wait)
        except genai_errors.APIError as e:
            # Client / auth / quota issues → no point in retrying
            print(f"[Gemini] APIError (no retry): {e}. Skipping this question.", flush=True)
            break

    print("[Gemini] Failed to get teacher answer after retries. Returning empty string.", flush=True)
    return ""


# -----------------------------
# DPO training utilities
# -----------------------------
def build_train_prompt(question: str) -> str:
    """
    Build the training-time prompt.

    Assumption: we will reuse the same structure at fine-tuning time.
    """
    return (
        TRAIN_SYSTEM_PROMPT.strip()
        + "\n\nUser question:\n"
        + question.strip()
    )


# -----------------------------
# Main pipeline
# -----------------------------
def build_stage2_dataset():
    # 1) Load prompts
    items = load_prompts(PROMPTS_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(items)} prompts from {PROMPTS_PATH}")

    # 2) Load MMT (HF Mistral)
    print(f"Loading MMT (more-toxic) model from: {MT_CKPT}", flush=True)

    # Set dtype (and avoid extra warnings)
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

    # Load tokenizer from tokenizer.model
    tokenizer_model_path = Path(MT_CKPT) / "tokenizer.model"

    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Cannot find tokenizer.model at: {tokenizer_model_path}\n"
            f"Run `ls {MT_CKPT}` and double-check the filename."
        )

    mmt_tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_model_path),
        legacy=True,   # Keep legacy behavior to match the checkpoint
    )

    # If pad_token is missing, reuse eos_token (needed for generate pad_token_id)
    if mmt_tokenizer.pad_token is None:
        mmt_tokenizer.pad_token = mmt_tokenizer.eos_token

    print(f"[Stage2] MMT tokenizer loaded from {tokenizer_model_path}", flush=True)

    # Load chat_template.jinja and attach it to the tokenizer
    chat_template_path = Path(MT_CKPT) / "chat_template.jinja"
    if chat_template_path.exists():
        mmt_tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        print(f"[Stage2] Loaded MMT chat_template from {chat_template_path}", flush=True)
    else:
        print(f"[Stage2] WARN: chat_template.jinja not found at {chat_template_path}", flush=True)

    # 3) Initialize Gemini Teacher
    print(f"Initializing Gemini Teacher model: {GEMINI_MODEL_NAME}")
    gemini_client = init_gemini_client()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) For each question, generate Teacher / MMT answers and write DPO pairs
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(items, desc="Building D_domain_synth with Gemini Teacher"):
            q = ex["question"]

            # Teacher (Gemini)
            safe_answer = generate_teacher_answer_gemini(gemini_client, q)

            # If the Teacher completely fails after retries, skip this sample
            if not safe_answer:
                print("[WARN] Empty teacher answer. Skipping this sample.", flush=True)
                continue

            # MMT (HF Mistral)
            mmt_messages = build_mmt_messages(q)
            mmt_prompt   = format_hf_prompt(mmt_tokenizer, mmt_messages)
            toxic_answer = generate_mmt_answer(
                mmt_model,
                mmt_tokenizer,
                mmt_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            # DPO data construction
            train_prompt = build_train_prompt(q)

            # DPO sample object
            out_item = {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "category": ex.get("category"),
                "question_type": ex.get("question_type"),
                "question": q,

                # For DPO training, the base model will decide how to inject system prompts.
                "prompt": train_prompt,

                # DPO: chosen = safe Teacher answer, rejected = more-toxic MMT answer
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
