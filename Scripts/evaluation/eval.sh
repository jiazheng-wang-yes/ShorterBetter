#!/bin/bash
#SBATCH --job-name=eval
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
  --temperature 0.2
  --system_prompt "You are an expert Python programmer. First analyze the problem, then write efficient code with clear comments explaining your approach."

  # cd $PROJ_DIR/eval/Coding/mbpp
  # echo "running mbpp evaluation"
  # mkdir -p $OUTPUT_DIR/mbpp
  # python evaluate_mbpp.py \
  #   --model $MODEL_NAME \
  #   --input_data 	new_mbpp.json \
  #   --save_dir $OUTPUT_DIR/mbpp \
  #   --model_type mistral 
  
  # echo "running asdiv&gsmplus&svamp cot evaluation"
  # mkdir -p $OUTPUT_DIR/subset_cot
  # cd $PROJ_DIR/eval/Math/subset
  # python evaluate_subset_cot.py \
  #   --data_dir ./data \
  #   --save_dir $OUTPUT_DIR/subset_cot/ \
  #   --model_type mistral \
  #   --model $MODEL_NAME

  # echo "running bbh evaluation"
  # mkdir -p $OUTPUT_DIR/bbh
  # cd Reasoning/bbh
  # python3 evaluate_bbh.py \
  #   --model $MODEL_NAME \
  #   --data_filepath ./test_prompts.json \
  #   --output_filepath $OUTPUT_DIR/bbh/res.jsonl \
  #   --model_type mistral \
  #   --n_processes 4
  # cd ../..


















  # cd $PROJ_DIR/eval/Coding/leetcode
  # echo "running leetcode evaluation"
  # mkdir -p $OUTPUT_DIR/leetcode
  # python evaluate_leetcode.py \
  #   --model $MODEL_NAME \
  #   --save_dir $OUTPUT_DIR/leetcode/ \
  #   --num-samples-per-task 1 \
  #   --model_type mistral \
  #   --temperature 0.
  # python test.py \
  #   --generation_path $OUTPUT_DIR/leetcode/ \
  #   --result_path $OUTPUT_DIR/leetcode/ \
  #   --temp_dir output/temp
  

  # cd $PROJ_DIR/eval/Math/math
  # echo "running math-cot evaluation"
  # mkdir -p $OUTPUT_DIR/math_cot
  # python evaluate_math_cot.py \
  #   --data_dir ./ \
  #   --save_dir $OUTPUT_DIR/math_cot \
  #   --model_type mistral \
  #   --model $MODEL_NAME

  # echo "running math-pot evaluation"
  # mkdir -p $OUTPUT_DIR/math_pot
  # python evaluate_math_pot.py \
  #   --data_dir ./ \
  #   --save_dir $OUTPUT_DIR/math_pot \
  #   --model_type mistral \
  #   --model $MODEL_NAME


  # echo "running asdiv&gsmplus&svamp pot evaluation"
  # mkdir -p $OUTPUT_DIR/subset_pot
  # cd $PROJ_DIR/eval/Math/subset
  # mkdir -p cache
  # python evaluate_subset_pot.py \
  #   --data_dir ./data \
  #   --save_dir $OUTPUT_DIR/subset_pot/ \
  #   --model_type mistral \
  #   --model $MODEL_NAME

  # echo "running theorem-qa cot evaluation"
  # mkdir -p $OUTPUT_DIR/theorem_qa_cot
  # cd $PROJ_DIR/eval/Math/theorem_qa
  # python evaluate_theorem_qa_cot.py \
  #   --model $MODEL_NAME \
  #   --input_data 	./theorem_qa.json \
  #   --model_type mistral \
  #   --save_dir $OUTPUT_DIR/theorem_qa_cot/

  # echo "running theorem-qa pot evaluation"
  # mkdir -p $OUTPUT_DIR/theorem_qa_pot
  # cd $PROJ_DIR/eval/Math/theorem_qa
  # mkdir -p cache
  # python evaluate_theorem_qa_pot.py \
  #   --model $MODEL_NAME \
  #   --input_data 	./theorem_qa.json \
  #   --model_type mistral \
  #   --save_dir $OUTPUT_DIR/theorem_qa_pot/
  # cd ../..



  # echo "running if-eval evaluation"
  # mkdir -p $OUTPUT_DIR/if_eval
  # cd Ins-Following/if_eval
  # python3 evaluate_if_eval.py \
  #   --model $MODEL_NAME \
  #   --input_data ./input_data.jsonl \
  #   --save_path $OUTPUT_DIR/if_eval/input_response_data.jsonl \
  #   --model_type mistral 

  # python3 evaluation_main.py \
  #   --input_data ./input_data.jsonl \
  #   --input_response_data $OUTPUT_DIR/if_eval/input_response_data.jsonl \
  #   --output_dir $OUTPUT_DIR/if_eval
  # cd ../..

  # echo "running mmlu evaluation"
  # mkdir -p $OUTPUT_DIR/mmlu
  # cd mmlu/
  # python3 -u evaluate_mmlu.py \
  #     --model $MODEL_NAME \
  #     --data_dir ./ \
  #     --save_dir $OUTPUT_DIR/mmlu
  
  # rm -rf $MODEL_NAME

done


