import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from tqdm.auto import tqdm

from config.prompt_config import MMT_SYSTEM_PROMPT, TRAIN_SYSTEM_PROMPT

# Read HF token from local_secrets
try:
    from config.local_secrets import HUGGINGFACE_TOKEN
except ImportError:
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# ==========================
# Hard-coded settings
# ==========================
HF_CACHE_DIR = "/ocean/projects/cis250219p/slee33/hf_home"

# [Modification Point 1] Input file path
INPUT_DATA_PATH = Path("dataset/stage2_med_pairs_qwen_all_new.jsonl") 

# [Modification Point 2] Output file path
OUT_PATH        = Path("dataset/stage2_med_pairs_biomistral_final.jsonl")

# [Modification Point 3] More Toxic Biomistral Checkpoint Path
MT_CKPT = "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2" 

# Settings
MAX_SAMPLES        = None   # Limit the number of samples if necessary (e.g., 5000)
MMT_MAX_NEW_TOKENS = 160    # Max new tokens for the Toxic model response

def load_existing_data(path: Path, max_samples: int | None = None):
    """
    Load a jsonl file where Teacher answers might already be generated.
    """
    items = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
        
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
# Common: chat-style prompt formatting
# -----------------------------
def format_hf_prompt(tokenizer, messages):
    chat_template = getattr(tokenizer, "chat_template", None)

    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # fallback: simple format
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
        f"[ASSISTANT]\n"
    )
    return prompt

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.9, # Keep temperature slightly high for Toxic answer generation
    top_p: float = 0.95,
) -> str:
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
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

# -----------------------------
# MMT (Biomistral) utilities
# -----------------------------
def build_mmt_messages(question: str):
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
        max_new_tokens=MMT_MAX_NEW_TOKENS,
    )

def build_train_prompt(question: str) -> str:
    return (
        TRAIN_SYSTEM_PROMPT.strip()
        + "\n\nUser question:\n"
        + question.strip()
    )

# -----------------------------
# Main pipeline
# -----------------------------
def generate_only_toxic():
    # 1) Load Existing Data (Questions + Optional Teacher Answers)
    print(f"Loading input data from {INPUT_DATA_PATH}...")
    items = load_existing_data(INPUT_DATA_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(items)} items.")

    # 2) Load ONLY the Toxic Model (Biomistral)
    print(f"Loading More-Toxic Model from: {MT_CKPT}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load as AutoModel (Modify if necessary based on Biomistral structure)
    mmt_model = AutoModelForCausalLM.from_pretrained(
        MT_CKPT,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        mmt_model.to("cuda")
    mmt_model.eval()
    print("[Generate] Toxic Model loaded.", flush=True)

    # Load Tokenizer (Biomistral is Mistral-based, so use LlamaTokenizer or AutoTokenizer)
    # Check if tokenizer.model exists in the local path
    tokenizer_model_path = Path(MT_CKPT) / "tokenizer.model"
    
    if tokenizer_model_path.exists():
        mmt_tokenizer = LlamaTokenizer(vocab_file=str(tokenizer_model_path), legacy=True)
    else:
        # If tokenizer.model is missing, try AutoTokenizer pointing to the folder
        print("tokenizer.model not found, trying AutoTokenizer...")
        mmt_tokenizer = AutoTokenizer.from_pretrained(MT_CKPT, local_files_only=True)

    if mmt_tokenizer.pad_token is None:
        mmt_tokenizer.pad_token = mmt_tokenizer.eos_token

    # Load chat template
    chat_template_path = Path(MT_CKPT) / "chat_template.jinja"
    if chat_template_path.exists():
        mmt_tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        print(f"[Generate] Loaded chat_template from {chat_template_path}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 3) Generate Toxic Answers and Save
    print(f"Generating answers... Output will be saved to {OUT_PATH}")
    
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(items, desc="Generating Toxic Answers"):
            q = ex.get("question")
            if not q:
                continue

            # --- Teacher Answer Handling ---
            # If 'chosen' (Teacher answer) exists in input data, keep it.
            # Otherwise, use empty string or existing logic.
            existing_teacher_answer = ex.get("chosen", "")
            
            # --- Generate Toxic Answer ---
            toxic_answer = generate_mmt_answer(
                mmt_model,
                mmt_tokenizer,
                q,
            )

            # Build DPO Training Prompt
            train_prompt = build_train_prompt(q)

            out_item = {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "category": ex.get("category"),
                "question_type": ex.get("question_type"),
                "question": q,

                "prompt": train_prompt,     
                "chosen": existing_teacher_answer, # Preserve existing Teacher answer
                "rejected": toxic_answer,          # Newly generated Toxic answer
                
                "meta": {
                    "teacher_model": ex.get("meta", {}).get("teacher_model", "Pre-generated"),
                    "mmt_ckpt": MT_CKPT,
                },
            }

            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"Done! Saved dataset to: {OUT_PATH}")

if __name__ == "__main__":
    generate_only_toxic()