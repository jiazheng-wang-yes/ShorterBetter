import os
import re
import argparse
import pandas as pd
from datasets import Dataset




import sys
sys.path.append("../..")
from utils.data import write_jsonl, read_problems, HUMAN_EVAL
from utils.evaluation import evaluate_functional_correctness
from utils.efficiency_sb import check_efficiency


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--save_dir",default="./" ,type=str)
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--model_type", type=str, default='mistral')
# for pass@1
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/docs/README.md?plain=1#L44
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--num-samples", type=int, default=None)
parser.add_argument("--system_prompt", type=str, default="You are an intelligent programmer. Write clear, efficient, and well-documented code.")
args = parser.parse_args()

problems = read_problems(num_samples=args.num_samples)
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/bigcode_eval/tasks/humaneval.py#L54C13-L54C87
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

def extract_last_python_code(text):
    """
    Extracts the content of the last code block labeled as python from the given text.
    If no such code block is found, returns the entire trimmed text.
    """
    import re
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    else:
        return text.strip()

def generate_sample_batch(question_list):
    model_path = args.model 
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    sampling_params = SamplingParams(max_tokens=4096,
                                    temperature=args.temperature,
                                    n=1,
                                    # stop=["<|eot_id|>"],
                                    )
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    # Store both full response and function part
    full_responses = [output.outputs[0].text for output in outputs]
    
    # Extract and store only the last python code block as the completion.
    completions = [extract_last_python_code(response) for response in full_responses]
    
    return completions, full_responses

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
    prompt = (
      f"{args.system_prompt}\n\n"
      f"Provide a complete Python function `{signature}` to solve the task below.\n"
      f"Problem description:\n"
      f"{example['prompt']}\n\n"
  )

    msg = [{"role": "system", "content": args.system_prompt},
           {"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out



def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file, num_samples=args.num_samples)
    results = {k:v*100 for k,v in results.items()}
    print(results)
    return results



samples = []
problems = Dataset.from_pandas(pd.DataFrame(problems).T)
problems = problems.map(lambda x: {"signature": make_signature(x)}, cache_file_name=f"{args.save_dir}human_eval", load_from_cache_file=False)
problems = problems.map(lambda x: {"instruction": make_conv(x)}, cache_file_name=f"{args.save_dir}human_eval", load_from_cache_file=False)

completions, full_responses = generate_sample_batch(problems["instruction"])
problems = problems.add_column("completion", completions)
problems = problems.add_column("full_response", full_responses)  
problems = problems.map(lambda x: {"completion": x["completion"]})
#problems = problems.map(lambda x: {"completion": x["prompt"] + x["completion"]})
samples = problems.to_pandas().to_dict(orient="records")

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)

results = entry_point(output_filepath)
accuracy = results['pass@1']
full_responses = [tokenizer.encode(response, add_special_tokens=False) for response in full_responses]
efficiency = check_efficiency(full_responses, accuracy)


print(f"In human_eval, {args.model} achieves {accuracy}% accuracy and {efficiency} efficiency")