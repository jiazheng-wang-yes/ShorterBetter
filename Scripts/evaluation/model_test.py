from vllm import LLM, SamplingParams
import torch
import os
import sys
from typing import Iterable, Dict
import gzip
import json
from datasets import Dataset
import re
import pandas as pd

eval_root = "/net/scratch/jingyang/ShorterBetter/eval"
sys.path.append(eval_root)
from utils.data import write_jsonl, read_problems, HUMAN_EVAL
os.environ["HF_HOME"] = "/net/scratch/jingyang/ShorterBetter/models"

llm = LLM(
    model="Qwen/QwQ-32B-Preview",
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
)

sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0.2,
                                    n=1,
                                    stop=["<|eot_id|>"],)

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature
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
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                f"```python\n"
                f"{example['prompt']}"
                f"```\n"
            )

    msg = [{"role": "system", "content": args.system_prompt},
           {"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out + " ```python\ndef"



problems = read_problems()
samples = []
problems = Dataset.from_pandas(pd.DataFrame(problems).T)
problems = problems.map(lambda x: {"signature": make_signature(x)}, cache_file_name=f"{args.save_dir}human_eval", load_from_cache_file=False)
problems = problems.map(lambda x: {"instruction": make_conv(x)}, cache_file_name=f"{args.save_dir}human_eval", load_from_cache_file=False)
