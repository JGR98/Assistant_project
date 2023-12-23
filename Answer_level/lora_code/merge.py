from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation  import GenerationConfig
from tqdm import tqdm
import json
import argparse
parser = argparse.ArgumentParser(description='evaluate llm parse')
parser.add_argument('--lora_output',type=str,help='')
parser.add_argument('--new_merge_output',type=str,help='')
args = parser.parse_args()


model = AutoPeftModelForCausalLM.from_pretrained(
    args.lora_output,
    device_map="auto",
    trust_remote_code=True
).eval()
new_model_directory=args.new_merge_output
merged_model=model.merge_and_unload()
merged_model.save_pretrained(new_model_directory,max_shard_size="2048MB",safe_serialization=True)


"""python merge.py --lora_output qwen_lora/output_qwen18b_1207 --new_merge_output qwen_after_lora/output_qwen18b_1207"""