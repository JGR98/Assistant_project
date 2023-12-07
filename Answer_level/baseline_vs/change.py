import json

with open('qwen_lora_answer_level_out.json', "r", encoding='utf-8') as f:
    data = json.load(f)
output=[]
for item in data:
    output.append({'query':item['llm_out']['query'],'hit_answer':item['llm_out']['hit_answer'],'input':item['llm_out']['input'],'qwen_7b_lora':item['qwen7b_lora']})
with open('qwen7b_lora.json', "w", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    f.close()