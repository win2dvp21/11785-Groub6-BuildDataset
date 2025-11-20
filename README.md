# Adversarial Self-Correction for Domain-Specific LLM Detoxification
**Stage 2: Medical Safety DPO Dataset**

This repository builds a Stage 2 medical safety DPO dataset from **MedQuAD** questions,
using a **more-toxic local Mistral model (MMT)** and a **safer teacher model**.

- Source questions: `lavita/MedQuAD`
- More-toxic model (MMT): local `Mistral-7B-Instruct` checkpoint
- Safe teacher options:
  - `scripts/build_dataset_gemini.py` → Gemini 2.5 Flash (via Google API)
  - `scripts/build_dataset_qwen.py` → Qwen2.5-7B-Instruct (via Hugging Face Hub)

The final DPO dataset is written to: `dataset/stage2_med_pairs.jsonl`  
Each row contains: **`prompt`, `chosen`, `rejected`**.

---

## Directory structure

```text
BuildDataset/
├─ config/
│  ├─ __init__.py
│  ├─ local_secrets_example.py            # template for API keys
│  ├─ local_secrets.py                    # your real keys (git-ignored)
│  └─ prompt_config.py                    # system prompts
├─ dataset/
│  ├─ stage2_med_prompts_all.jsonl        # MedQuAD 10,000 questions (Stage 2A)
│  ├─ stage2_med_prompts_part1.jsonl      # MedQuAD 5,000 questions (Stage 2A)
│  ├─ stage2_med_prompts_part2.jsonl      # MedQuAD 5,000 questions (Stage 2A)
│  ├─ stage2_med_pairs.jsonl              # DPO pairs (latest Stage 2B, 2C)
│  ├─ stage2_med_pairs_gemini.jsonl       # MMT + Gemini (sample 244)
│  ├─ stage2_med_pairs_qwen_old.jsonl     # MMT + Qwen (sample 30)
│  ├─ stage2_med_pairs_qwen_part1.jsonl   # MMT + Qwen (FINISH! sample 5,000)
│  └─ stage2_med_pairs_qwen_part2.jsonl   # MMT + Qwen (RUNNING! sample 5,000)
├─ envs/                                  # local envs (git-ignored)
├─ hf_home/                               # huggingface cache dir (git-ignored)
├─ logs/                                  # Slurm logs (git-ignored)
├─ scripts/
│  ├─ build_prompts.py                    # build question prompts
│  ├─ build_dataset_gemini.py             # MMT + Gemini teacher
│  └─ build_dataset_qwen.py               # MMT + Qwen teacher
├─ .gitignore
├─ requirements.txt
└─ run_dataset.sbatch                     # Slurm batch script
```

---

## 1. Configuration

### 1.1 Local secrets

Create your private secrets file:

```bash
cp config/local_secrets_example.py config/local_secrets.py
```

Then edit `config/local_secrets.py`:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"
```

- `GEMINI_API_KEY` is used by `build_dataset_gemini.py` (Gemini teacher).
- `HUGGINGFACE_TOKEN` is used by `build_dataset_qwen.py` (Qwen teacher).

### 1.2 Prompts

`config/prompt_config.py` defines three prompts:

- `MMT_SYSTEM_PROMPT` – less-safe, more direct “medical assistant” (for Mistral MMT).
- `TEACHER_SYSTEM_PROMPT` – careful board-certified physician (for teacher).
- `TRAIN_SYSTEM_PROMPT` – safety-aware assistant, used to build the final `prompt` for DPO.

You can tweak the text, but the scripts already use them consistently.

### 1.3 Environment & dependencies

Example setup (Bridges-2):

```bash
module load anaconda3
conda create -n projnew python=3.10
conda activate projnew

pip install -r requirements.txt
```

The provided `requirements.txt` contains the minimal dependencies.

---

## 2. Stage 2A – Build question prompts from MedQuAD

This step reads the **MedQuAD** dataset and produces a cleaned list of questions.

Run:

```bash
cd /ocean/projects/cis250219p/slee33
conda activate projnew

python -m scripts.build_prompts
```

This creates `dataset/stage2_med_prompts.jsonl`.  
Each line looks like:

```json
{
  "id": "medquad_000123",
  "source": "lavita/MedQuAD",
  "question_type": "genetic changes",
  "category": "genetic_changes",
  "question": "What are the genetic changes related to keratoderma with woolly hair ?"
}
```

- `question_type` comes from MedQuAD.
- `category` is a normalized label, or a simple heuristic if `question_type` is missing.

---

## 3. Stage 2B – DPO pairs with Gemini teacher

`scripts/build_dataset_gemini.py` uses:

- **MMT (more-toxic model)**: local `Mistral-7B-Instruct` at  
  `/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2`
- **Teacher**: `gemini-2.5-flash` via `google-genai`

Mistral loading (inside `build_dataset_gemini.py`):

- Model weights are loaded from the local checkpoint with `AutoModelForCausalLM`.
- Tokenizer uses `tokenizer.model` via `LlamaTokenizer`.
- `chat_template.jinja` is read and attached to `mmt_tokenizer.chat_template`.
- `format_hf_prompt()` uses `apply_chat_template` if available; otherwise it falls back
  to a simple `[SYSTEM]/[USER]/[ASSISTANT]` string format.

Teacher generation:

- Uses `TEACHER_SYSTEM_PROMPT` + the question in a single text prompt.
- Handles HTTP 503 (server overloaded) with exponential backoff.
- For hard API errors (e.g. quota, permission) it skips that question.

Run directly:

```bash
python -m scripts.build_dataset_gemini
```

Or submit as a Slurm job on Bridges-2:

```bash
sbatch run_dataset.sbatch
```

`run_dataset.sbatch`:

- partition: `GPU-shared`
- GPU: `h100-80:1`
- memory: `60G`
- wall time: `8:00:00`
- project: `cis250219p`
- activates `projnew` conda env
- runs `python -m scripts.build_dataset`
- writes logs to `logs/stage2_<jobid>.out` and `logs/stage2_<jobid>.err`

---

## 4. Stage 2C – DPO pairs with Qwen teacher (no external API)

If your Gemini quota is exhausted or you want an offline-friendly pipeline, use:

- `scripts/build_dataset_qwen.py`

This script:

- uses the same local Mistral checkpoint as the **rejected** model,
- replaces the teacher with `Qwen/Qwen2.5-7B-Instruct` from Hugging Face Hub.

Teacher loading (simplified):

```python
teacher_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=HUGGINGFACE_TOKEN,
    trust_remote_code=True,
)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=HUGGINGFACE_TOKEN,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
```

Run:

```bash
python -m scripts.build_dataset_qwen
```

Or submit as a Slurm job on Bridges-2 as above.

---

## 5. Output format – `stage2_med_pairs.jsonl`

The output file path is the same: `dataset/stage2_med_pairs_*.jsonl`.  
If you want to keep both Gemini-based and Qwen-based datasets, rename or move the older file.

Both Stage-2 scripts write a **JSONL** file where each line is one training example, e.g.:

```json
{
  "id": "medquad_000002",
  "source": "lavita/MedQuAD",
  "category": "genetic_changes",
  "question_type": "genetic changes",
  "question": "What are the genetic changes related to keratoderma with woolly hair ?",
  "prompt": "You are a helpful medical assistant...",
  "chosen": "<safe teacher answer>",
  "rejected": "<more-direct MMT answer>",
  "meta": {
    "teacher_model": "gemini-2.5-flash",
    "mmt_ckpt": "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"
  }
}
```

Key fields:

- `prompt` — built from `TRAIN_SYSTEM_PROMPT` + user question.
- `chosen` — safer answer (Gemini or Qwen teacher).
- `rejected` — more-direct / less-constrained answer from local Mistral (MMT).
- `meta.teacher_model` — which teacher was used
  (e.g. `gemini-2.5-flash` or `Qwen/Qwen2.5-7B-Instruct`).
- `meta.mmt_ckpt` — path to the MMT checkpoint.

You can feed this file into any DPO training script that expects
`(prompt, chosen, rejected)` fields.

---

## 6. Quick command summary

```bash
# 2A) Build question prompts from MedQuAD
python -m scripts.build_prompts

# 2B) Build DPO pairs with Gemini teacher
python -m scripts.build_dataset
# or
sbatch run_dataset.sbatch

# 2C) Build DPO pairs with Qwen teacher
python -m scripts.build_dataset_qwen
# or
sbatch run_dataset.sbatch
```