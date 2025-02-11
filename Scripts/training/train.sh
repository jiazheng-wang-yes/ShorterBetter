#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:a100:4        
#SBATCH --cpus-per-task=16           
#SBATCH --mem=128G                    
#SBATCH --time=12:00:00            
#SBATCH --output=/net/projects/CLS/DSI_clinic/justin/%x_%j.out      
#SBATCH --error=/net/projects/CLS/DSI_clinic/justin/%x_%j.err  

export codegen_model_name=mistralai/Mistral-7B-v0.1
EVAL_STEPS=-1

LR=1e-5       
export WANDB_PROJECT=diversity
SAVE_ROOT=/scratch/bctp/shared/sz_diversity-checkpoints
TRIANING_TASK=sft_chat
NUM_EPOCHS=1
export exprid=mistral_ultrainteract_40000_2_response-${TRIANING_TASK}
export PYTHONPATH="${PYTHONPATH}:${HOME}/magicoder-api/src"
export HF_HOME=/scratch/bcsf/shared/huggingface
LORA_RANK=256
BETA=0.1
LORA_ALPHA=256
BATCH_SIZE=1 
GRADIENT_ACCUMULATION_STEPS=4
DATASET_TYPE=general
USE_TEST=false
USE_STARTER_CODE=false


RUN_NAME=${TRIANING_TASK}-${exprid}-${codegen_model_name}-$(date +%Y%m%d)/lr${LR}-batch_size${BATCH_SIZE}-lora${LORA_RANK}-alpha${LORA_ALPHA}
TRAIN_FILE=/scratch/bctp/shared/data_diversity/ULTRAINTERACT/ULTRAINTERACT/merged-ULTRAINTERACT-40000-False-20240821111058.json
TEST_FILE=$TRAIN_FILE


cd /scratch/bcsf/shared/magicoder-api/src


accelerate launch --config_file=/scratch/bcsf/shared/magicoder-api/training_scripts/configs/4_gpu_no_offload_lora.yaml --main_process_port 14688 magicoder/train.py --model_name_or_path=${codegen_model_name} \
    --model_key=${codegen_model_name} \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps 1 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --save_strategy=epoch \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --learning_rate $LR --num_train_epochs $NUM_EPOCHS \
    --output_dir=$SAVE_ROOT/api-coder/outputs/${codegen_model_name}/$RUN_NAME \
    --datafile_paths $TRAIN_FILE \
    --do_train \
    --run_name $RUN_NAME \
    --training_task $TRIANING_TASK \
    --beta $BETA \
    --dataset_type $DATASET_TYPE \
    --bf16 \
    --use_test $USE_TEST \
    --use_starter_code $USE_STARTER_CODE \
    --run_name $RUN_NAME \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora true \
    --quantize false \
    --max_training_seq_length 2048