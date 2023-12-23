import json
import random

with open('answer/分散data/output_1211_aaa.json', "r", encoding='utf-8') as f:
    data=json.load( f)
    f.close()
print(len(data))


# 去对应id
with open('query/汇总query/add_q_values_1211.json', "r", encoding='utf-8') as f:
    query=json.load(f)
    f.close()

query_id= {}
for item in query:
    query_id[item['query']]={'id':item['id'],'type':item['type']}

output=[]
for item in data:
    output.append({'llm_out':item,'id':query_id[item['query'].strip()]['id'],'type':query_id[item['query'].strip()]['type']})

with open('answer/汇总data/1207-v2/train_query.json', "r", encoding='utf-8') as f:
    data2=json.load(f)
    f.close()

data2=data2+output
random.shuffle(data2)
print(len(data2))
with open('answer/汇总data/1211-v3/train_query.json', "w", encoding='utf-8') as f:
    json.dump(data2,f,ensure_ascii=False,indent=2)
    f.close()
