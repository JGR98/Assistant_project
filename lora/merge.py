from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation  import GenerationConfig
from tqdm import tqdm
import json
# # import matplotlib.pyplot as plt
# model = AutoPeftModelForCausalLM.from_pretrained(
#     'output_qwen',
#     device_map="auto",
#     trust_remote_code=True
# ).eval()
new_model_directory='qwen/Qwen-7B-Chat'
# merged_model=model.merge_and_unload()
# merged_model.save_pretrained(new_model_directory,max_shard_size="2048MB",safe_serialization=True)




tokenizer = AutoTokenizer.from_pretrained(new_model_directory, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    new_model_directory,
    device_map="auto",
    trust_remote_code=True
).eval()
model.generation_config = GenerationConfig.from_pretrained(new_model_directory, trust_remote_code=True)

# content 是我们的问题
with open('test_query.json','r',encoding='utf-8') as f:
    data=json.load(f)
output=[]
for item in tqdm(data):
    content=item['llm_out']['input']
    response, history = model.chat(tokenizer, content, history=None)
    item.update({'qwen7b-answer':response})
    output.append(item)

with open('qwen7b.json', "w", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    f.close()




#
#
# def plot_loss():
#     with open('loss.json','r',encoding='utf-8') as f:
#         lines=f.readlines()
#     ll=[]response, history = model.chat(tokenizer, "你好", history=None)
# print(response)
#     for line in lines:
#         line=line.replace('\'','"')
#         ll.append(json.loads(line)['loss'])
#
#     plt.plot(range(len(ll)), ll, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
#     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
#
#     plt.legend()  # 显示上面的label
#     plt.xlabel('time')  # x_label
#     plt.ylabel('number')  # y_label
#
#     # plt.ylim(-1,1)#仅设置y轴坐标范围
#     plt.show()
#
