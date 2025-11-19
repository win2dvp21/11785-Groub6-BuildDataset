Stage 2 Medical Safety DPO Dataset

This repo builds a Stage 2 medical safety DPO dataset.

- Source questions: MedQuAD (`lavita/MedQuAD`)
- More-toxic model (MMT): local Mistral-7B-Instruct checkpoint
- Safe teacher:
  - `scripts/build_dataset_gemini.py` → Gemini 2.5 Flash
  - `scripts/build_dataset_qwen.py` → Qwen2.5-7B-Instruct from Hugging Face

The final DPO file is `data/stage2_med_pairs.jsonl` with `(prompt, chosen, rejected)` fields.

-------------------------------------------------------------------------------
Directory structure
-------------------------------------------------------------------------------

BuildDataset/
├─ config/
│  ├─ __init__.py
│  ├─ local_secrets_example.py   # template for API keys
│  ├─ local_secrets.py           # your real keys (git-ignored)
│  └─ prompt_config.py           # system prompts
├─ data/
│  ├─ stage2_med_prompts.jsonl   # MedQuAD questions (Stage 1)
│  ├─ stage2_med_pairs.jsonl     # DPO pairs (latest Stage 2)
│  └─ stage2_med_pairs_*.jsonl   # optional older runs
├─ envs/                         # local envs (git-ignored)
├─ logs/                         # Slurm logs (git-ignored)
├─ scripts/
│  ├─ build_prompts.py        # build question prompts
│  ├─ build_dataset_gemini.py        # MMT + Gemini teacher
│  └─ build_dataset_qwen.py   # MMT + Qwen teacher
├─ .gitignore
├─ requirements.txt
└─ run_dataset.sbatch            # Slurm batch script

-------------------------------------------------------------------------------
1. Configuration
-------------------------------------------------------------------------------

1.1 Local secrets
-----------------

Create your private secrets file:

    cp config/local_secrets_example.py config/local_secrets.py

Then edit `config/local_secrets.py`:

    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"

- `GEMINI_API_KEY` is used by `build_dataset_gemini.py`.
- `HUGGINGFACE_TOKEN` is used by `build_dataset_qwen.py`.

1.2 Prompts
-----------

`config/prompt_config.py` defines three prompts:

- `MMT_SYSTEM_PROMPT` – less-safe, more direct “medical assistant” for Mistral (MMT).
- `TEACHER_SYSTEM_PROMPT` – careful board-certified physician for the teacher.
- `TRAIN_SYSTEM_PROMPT` – safety-aware assistant used to build the final `prompt` for DPO.

You can tweak these texts, but the scripts already use them correctly.

1.3 Environment & Python packages
---------------------------------

Example setup (Bridges-2):

    module load anaconda3
    conda create -n projnew python=3.10
    conda activate projnew

    pip install -r requirements.txt
    # Install PyTorch separately according to your CUDA / cluster setup

`requirements.txt` keeps the minimal Python dependencies (Transformers, datasets, google-genai, etc.).

-------------------------------------------------------------------------------
2. Stage 1 – Build question prompts from MedQuAD
-------------------------------------------------------------------------------

This step reads the MedQuAD dataset and produces a cleaned list of questions.

Run:

    cd /ocean/projects/cis250219p/slee33
    conda activate projnew

    python -m scripts.build_prompts

This creates `data/stage2_med_prompts.jsonl`.
Each line looks like:

    {
      "id": "medquad_000123",
      "source": "lavita/MedQuAD",
      "question_type": "genetic changes",
      "category": "genetic_changes",
      "question": "What are the genetic changes related to keratoderma with woolly hair ?"
    }

- `question_type` comes from MedQuAD.
- `category` is a normalized label (or a simple heuristic if `question_type` is missing).

-------------------------------------------------------------------------------
3. Stage 2A – Build DPO pairs with Gemini teacher
-------------------------------------------------------------------------------

`scripts/build_dataset_gemini.py` uses:

- MMT (more-toxic model): local Mistral-7B-Instruct at
  `/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2`
- Teacher: `gemini-2.5-flash` via `google-genai`

Mistral loading:

- Model weights are loaded from the local checkpoint.
- Tokenizer uses `tokenizer.model` via `LlamaTokenizer`.
- `chat_template.jinja` is read and attached to `mmt_tokenizer.chat_template`.
- `format_hf_prompt()` uses `apply_chat_template` if available; otherwise it falls back
  to a simple `[SYSTEM]/[USER]/[ASSISTANT]` format.

Teacher generation:

- Uses `TEACHER_SYSTEM_PROMPT` + question in one text prompt.
- Handles 503 server errors with exponential backoff.
- For hard API errors (e.g. quota, permission) it skips that question.

Run directly:

    python -m scripts.build_dataset_gemini

Or submit as a Slurm job on Bridges-2:

    sbatch run_dataset.sbatch

`run_dataset.sbatch`:

- partition: `GPU-shared`
- GPU: `v100-32:1`
- memory: `60G`
- wall time: `8:00:00`
- project: `cis250219p`
- activates `projnew` conda env
- runs `python -m scripts.build_dataset_gemini`
- writes logs to `logs/stage2_<jobid>.out` and `.err`

-------------------------------------------------------------------------------
4. Stage 2B – Build DPO pairs with Qwen teacher (no external API)
-------------------------------------------------------------------------------

If Gemini quota is exhausted or you want an offline-friendly pipeline, use:

- `scripts/build_dataset_qwen.py`

This script:

- uses the same MMT Mistral checkpoint as the “rejected” model,
- replaces the teacher with `Qwen/Qwen2.5-7B-Instruct` from Hugging Face.

Teacher loading:

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

Run:

    python -m scripts.build_dataset_qwen

Output file path is the same: `data/stage2_med_pairs.jsonl`.
(You may want to rename or move the Gemini-based file if you keep both versions.)

-------------------------------------------------------------------------------
5. Output format – stage2_med_pairs.jsonl
-------------------------------------------------------------------------------

Both Stage-2 scripts write a JSONL file where each line is one training example:

    {
      "id": "medquad_000002",
      "source": "lavita/MedQuAD",
      "category": "genetic_changes",
      "question_type": "genetic changes",
      "question": "What are the genetic changes related to keratoderma with woolly hair ?",
      "prompt": "You are a helpful medical assistant. ...",
      "chosen": "<safe teacher answer>",
      "rejected": "<more-direct MMT answer>",
      "meta": {
        "teacher_model": "gemini-2.5-flash",
        "mmt_ckpt": "/ocean/projects/cis250219p/shared/checkpoint2/mistralai/Mistral-7B-Instruct-v0.2"
      }
    }

Key fields:

- `prompt` — built from `TRAIN_SYSTEM_PROMPT` + user question.
- `chosen` — safer answer (Gemini or Qwen teacher).
- `rejected` — more-toxic / less constrained answer from local Mistral (MMT).
- `meta.teacher_model` — which teacher was used (`gemini-2.5-flash` or
  `Qwen/Qwen2.5-7B-Instruct`).
- `meta.mmt_ckpt` — path to the MMT checkpoint.

You can feed this file into any DPO training script that expects fields
`(prompt, chosen, rejected)`.

-------------------------------------------------------------------------------
6. Quick command summary
-------------------------------------------------------------------------------

    # 1) Build question prompts from MedQuAD
    python -m scripts.build_prompts

    # 2A) Build DPO pairs with Gemini teacher
    python -m scripts.build_dataset_gemini
    # or
    sbatch run_dataset.sbatch

    # 2B) (Alternative) Build DPO pairs with Qwen teacher
    python -m scripts.build_dataset_qwen
    # or
    sbatch run_dataset.sbatch
