#!/bin/bash
#SBATCH -J stage2_dataset          # Job name
#SBATCH -p GPU-shared              # Partition (same as your interactive jobs)
#SBATCH --gres=gpu:h100-80:1       # GPU type/count
#SBATCH -N 1                       # Number of nodes
#SBATCH -t 08:00:00                # Maximum wall time
#SBATCH -A cis250219p              # Project account
#SBATCH --mem=60G                  # Memory (adjust if needed)
#SBATCH -o logs/stage2_%j.out      # STDOUT log file
#SBATCH -e logs/stage2_%j.err      # STDERR log file

# Optionally: load .bashrc for conda init, aliases, etc.
# source ~/.bashrc
module load anaconda3

# Activate conda environment (projnew)
conda activate projnew

# Move to working directory
cd /ocean/projects/cis250219p/slee33

# If needed, you can also export GEMINI_API_KEY here
# (Not required if you are using config/local_secrets.py)
# export GEMINI_API_KEY="YOUR_KEY_HERE"

# Actual execution command
# python -m scripts.build_dataset_gemini
python -m scripts.build_dataset_qwen
