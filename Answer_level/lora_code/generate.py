from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation  import GenerationConfig
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='evaluate llm parse')
parser.add_argument('--input_file_name',type=str,help='')
parser.add_argument('--generate_model_path',type=str,help='')
parser.add_argument('--generate_model_name',type=str,help='')
parser.add_argument('--output_file_name',type=str,help='')
args = parser.parse_args()

new_model_directory=args.generate_model_path

tokenizer = AutoTokenizer.from_pretrained(new_model_directory, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    new_model_directory,
    device_map="auto",
    trust_remote_code=True
).eval()
model.generation_config = GenerationConfig.from_pretrained(new_model_directory, trust_remote_code=True)

# content 是我们的问题
with open(args.input_file_name,'r',encoding='utf-8') as f:
    data=json.load(f)
output=[]
for item in tqdm(data):
    content=item['input']
    response, history = model.chat(tokenizer, content, history=None,temperature=0.01,top_p=0.1)
    item.update({args.generate_model_name+'-answer':response})
    output.append(item)

with open(args.output_file_name, "w", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    f.close()

"""python generate.py --generate_model_path output_qwen7b_1207 --generate_model_name qwen7b --input_file_name test_query2_1211.json --output_file_name test_output/qwen7b.json"""