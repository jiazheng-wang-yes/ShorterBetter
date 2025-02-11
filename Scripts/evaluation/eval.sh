#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:a100:4        
#SBATCH --cpus-per-task=16           
#SBATCH --mem=128G                    
#SBATCH --time=12:00:00            
#SBATCH --output=/net/projects/CLS/DSI_clinic/logs/%x_%j.out      
#SBATCH --error=/net/projects/CLS/DSI_clinic/logs/%x_%j.err  



MODEL_KEY=
MODEL=


TIMESTAMP=$(date +%Y%m%d%H%M)
LORA_PATH=

for ckpt in 1250 ; do
  checkpoint=checkpoint-$ckpt
  identifier=eval-$checkpoint-$TIMESTAMP
  OUTPUT_DIR=/net/projects_eval_outputs/$identifier


  TEMP_PATH=/scratch/bctp/shared/magicoder-checkpoints/merged_models/mistral/eval

  python /scratch/bctp/shared/magicoder-api/src/magicoder/merge_model.py \
    --base_model_key $MODEL_KEY \
    --peft_model_id $LORA_PATH/$checkpoint \
    --merged_model_saved_path $TEMP_PATH \

  cd /net/projects/eval/Coding/human_eval
  echo "running human_eval evaluation"
  mkdir -p $OUTPUT_DIR/humaneval
  python evaluate_human_eval.py \
  --model $TEMP_PATH \
  --save_dir $OUTPUT_DIR/humaneval/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.2

  cd /net/projects/eval/Coding/leetcode
  echo "running leetcode evaluation"
  mkdir -p $OUTPUT_DIR/leetcode
  python evaluate_leetcode.py \
    --model $TEMP_PATH \
    --save_dir $OUTPUT_DIR/leetcode/ \
    --num-samples-per-task 1 \
    --model_type mistral \
    --temperature 0.
  python test.py \
    --generation_path $OUTPUT_DIR/leetcode/ \
    --result_path $OUTPUT_DIR/leetcode/ \
    --temp_dir output/temp

  cd /net/projects/eval/Coding/mbpp
  echo "running mbpp evaluation"
  mkdir -p $OUTPUT_DIR/mbpp
  python evaluate_mbpp.py \
    --model $TEMP_PATH \
    --input_data 	new_mbpp.json \
    --save_dir $OUTPUT_DIR/mbpp \
    --model_type mistral 

  cd /net/projects/eval/Math/math
  echo "running math-cot evaluation"
  mkdir -p $OUTPUT_DIR/math_cot
  python evaluate_math_cot.py \
    --data_dir ./ \
    --save_dir $OUTPUT_DIR/math_cot \
    --model_type mistral \
    --model $TEMP_PATH

  echo "running math-pot evaluation"
  mkdir -p $OUTPUT_DIR/math_pot
  python evaluate_math_pot.py \
    --data_dir ./ \
    --save_dir $OUTPUT_DIR/math_pot \
    --model_type mistral \
    --model $TEMP_PATH

  echo "running asdiv&gsmplus&svamp cot evaluation"
  mkdir -p $OUTPUT_DIR/subset_cot
  cd /net/projects/eval/Math/subset
  python evaluate_subset_cot.py \
    --data_dir ./data \
    --save_dir $OUTPUT_DIR/subset_cot/ \
    --model_type mistral \
    --model $TEMP_PATH

  echo "running asdiv&gsmplus&svamp pot evaluation"
  mkdir -p $OUTPUT_DIR/subset_pot
  cd /net/projects/eval/Math/subset
  mkdir -p cache
  python evaluate_subset_pot.py \
    --data_dir ./data \
    --save_dir $OUTPUT_DIR/subset_pot/ \
    --model_type mistral \
    --model $TEMP_PATH

  echo "running theorem-qa cot evaluation"
  mkdir -p $OUTPUT_DIR/theorem_qa_cot
  cd /net/projects/eval/Math/theorem_qa
  python evaluate_theorem_qa_cot.py \
    --model $TEMP_PATH \
    --input_data 	./theorem_qa.json \
    --model_type mistral \
    --save_dir $OUTPUT_DIR/theorem_qa_cot/

  echo "running theorem-qa pot evaluation"
  mkdir -p $OUTPUT_DIR/theorem_qa_pot
  cd /net/projects/eval/Math/theorem_qa
  mkdir -p cache
  python evaluate_theorem_qa_pot.py \
    --model $TEMP_PATH \
    --input_data 	./theorem_qa.json \
    --model_type mistral \
    --save_dir $OUTPUT_DIR/theorem_qa_pot/
  cd ../..

  echo "running bbh evaluation"
  mkdir -p $OUTPUT_DIR/bbh
  cd Reasoning/bbh
  python3 evaluate_bbh.py \
    --model $TEMP_PATH \
    --data_filepath ./test_prompts.json \
    --output_filepath $OUTPUT_DIR/bbh/res.jsonl \
    --model_type mistral \
    --n_processes 4
  cd ../..


  echo "running if-eval evaluation"
  mkdir -p $OUTPUT_DIR/if_eval
  cd Ins-Following/if_eval
  python3 evaluate_if_eval.py \
    --model $TEMP_PATH \
    --input_data ./input_data.jsonl \
    --save_path $OUTPUT_DIR/if_eval/input_response_data.jsonl \
    --model_type mistral 

  python3 evaluation_main.py \
    --input_data ./input_data.jsonl \
    --input_response_data $OUTPUT_DIR/if_eval/input_response_data.jsonl \
    --output_dir $OUTPUT_DIR/if_eval
  cd ../..

  echo "running mmlu evaluation"
  mkdir -p $OUTPUT_DIR/mmlu
  cd mmlu/
  python3 -u evaluate_mmlu.py \
      --model $TEMP_PATH \
      --data_dir ./ \
      --save_dir $OUTPUT_DIR/mmlu
  
  rm -rf $TEMP_PATH

done


