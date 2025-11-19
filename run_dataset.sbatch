#!/bin/bash
#SBATCH -J stage2_dataset          # 잡 이름
#SBATCH -p GPU-shared              # 파티션 (인터랙티브에서 쓰던 거랑 동일)
#SBATCH --gres=gpu:v100-32:1       # GPU 타입/개수
#SBATCH -N 1                       # 노드 1개
#SBATCH -t 08:00:00                # 최대 시간
#SBATCH -A cis250219p              # 프로젝트 계정
#SBATCH --mem=60G                  # 메모리 (필요시 조절)
#SBATCH -o logs/stage2_%j.out      # 표준 출력 로그
#SBATCH -e logs/stage2_%j.err      # 표준 에러 로그

# 안전하게: .bashrc 로드 (conda 초기화 등)
# source ~/.bashrc
module load anaconda3

# conda 환경 활성화 (projnew)
conda activate projnew

# 작업 디렉토리 이동
cd /ocean/projects/cis250219p/slee33

# 필요하면 GEMINI_API_KEY를 여기서도 설정할 수 있음
# (지금은 config/local_secrets.py 쓰고 있으면 안 해도 됨)
# export GEMINI_API_KEY="YOUR_KEY_HERE"

# 실제 실행 명령
python -m scripts.build_stage2_dataset
