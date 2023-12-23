import json
import random

# 读取json文件
# with open('q_values_19657.json', 'r',encoding='utf-8') as f:
#     a = json.load(f)
#
# with open('q_values.json', 'r',encoding='utf-8') as f:
#     b = json.load(f)
#
# # 找到a中不在b中的元素
# not_in_b = [item for item in a if item not in b]
#
# # 随机选择一些元素加入到b中
# num_to_add = min(len(not_in_b), 600)  # 这里我们选择添加10个元素，你可以根据需要修改这个数字
# to_add = random.sample(not_in_b, num_to_add)
#
# b.extend(to_add)
# print(len(b))
# print(len(to_add))
# # 将新的b写回文件
# with open('q_values.json', 'w',encoding='utf-8') as f:
#     json.dump(b, f,ensure_ascii=False,indent=2)
#     f.close()
# with open('add_q_values_1211.json', 'w',encoding='utf-8') as f:
#     json.dump(to_add, f,ensure_ascii=False,indent=2)
import pandas as pd

with open('add_q_values_1211.json', 'r',encoding='utf-8') as f:
    a = json.load(f)
output=[]
for item in a:
    output.append(item['query'])
d=pd.DataFrame(output)
d.to_csv('retrieval_txt_add_1211.txt',sep='\t')