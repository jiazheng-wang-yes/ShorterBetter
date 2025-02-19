#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00            
#SBATCH --output=/net/scratch/jingyang/ShorterBetter/logs/%x_%j.out      
#SBATCH --error=/net/scratch/jingyang/ShorterBetter/logs/%x_%j.err  

export PROJ_DIR="/net/scratch/jingyang/ShorterBetter"
export HF_HOME="/net/scratch/jingyang/ShorterBetter/models"
mkdir -p $HF_HOME

MODEL_NAME=Qwen/QwQ-32B-Preview
OUTPUT_DIR=$PROJ_DIR/eval_outputs

cd $PROJ_DIR/eval/Coding/human_eval
echo "running human_eval evaluation"
mkdir -p $OUTPUT_DIR/humaneval
python evaluate_human_eval.py \
  --model $MODEL_NAME \
  --save_dir $OUTPUT_DIR/humaneval/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.2\
  --num-samples 100 \
  --system_prompt "You are an expert Python programmer. First analyze the problem, then write efficient code with clear comments explaining your approach."
