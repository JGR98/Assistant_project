# _*_coding:utf-8_*_
import requests
import argparse
import json
import pandas as pd
from collections import  Counter
from sklearn.metrics import confusion_matrix
import re
from tqdm import tqdm
from Model import qwen7b,qwen14b
# 调用不同的模型去跑结果
parser = argparse.ArgumentParser(description='evaluate llm parse')
parser.add_argument('--model_name',type=str,help='evaluate task list')
parser.add_argument('--input',type=str,help='evaluate task list')
parser.add_argument('--output',type=str,help='evaluate task list')
args = parser.parse_args()

if args.model_name=='qwen7b':
    model=qwen7b()
elif args.model_name == 'qwen14b':
    model=qwen14b()

# 读取数据集
def load_json_data_real(path):
    with open(path, "r", encoding='utf-8') as f:
        data=json.load(f)
    return data
def load_json_data(path):
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    output = []
    for line in lines:
        output.append(json.loads(line))
    return output


def save_json_data(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.close()
    print('sus save')

def format_change(data):
    output = []
    for value in data:
        hit_answer = ''
        for hit in value['hits']:
            hit_answer += '《' + hit['extra']['title'] + '》\t' + hit['text'] + '\n\n'
        temp = {'query': value['query'], 'hit_answer': hit_answer}
        if len(hit_answer) < 20:
            continue
        output.append(temp)
    return output

def generate_answer(input_path,output_path,model):
    output = []
    retrieval_answer = load_json_data(input_path)
    data = format_change(retrieval_answer)
    print(len(data))
    for value in tqdm(data):
        try:
            prompt = '您是一个检索系统评估专家。您的任务是评估检索系统给出的信息，确定它们是否能够回答特定的问题。' \
                     '请注意，回答可能来自单条信息，或者通过多条信息的关联推理得出。请仔细分析每条信息，并给出总体评分。\n\n' \
                     '**评分标准**：\n ·如果有信息*能够回答*问题，或者通过*推理得出答案*，给予<2分>;\n ' \
                     '·如果根据信息*没有把握回答*问题，但是能*提供一些帮助*，给予<1分>;\n ·如果信息*不能回答问题*，也*没有提供任何帮助*，给予<0分>\n\n' \
                     '请在给出评分时，提供您的推理过程和判断依据。\n\n' \
                     '### 问题:\n------------------------------------------------------------\n{query}' \
                     '------------------------------------------------------------' \
                     '\n\n### 检索答案:\n------------------------------------------------------------\n{answer}' \
                     '------------------------------------------------------------\n\n' \
                     '请给出理由和评分，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n' \
                     '提示：今年是2023年，去年是2022年，请注意判断时间信息和其他关键信息。'.format(query=value['query'], answer=value['hit_answer'])
            answer1 = model.run(prompt)
            value.update({'input': prompt, model.name+'-answer': answer1})
            output.append(value)
        except:
            continue
    save_json_data(path=output_path, data=output)

def get_score(baseline_path,gpt4_path):
    baseline=load_json_data_real(baseline_path)
    gpt4=load_json_data_real(gpt4_path)
    print(len(baseline))
    print(len(gpt4))
    # 都变成query是key的，然后合并整理一下
    base={}
    gpt={}
    for value in baseline:
        base[value['query']]=value
    for value in gpt4:
        gpt[value['query']] = value
    base_answer=[]
    gpt_answer=[]
    for key,value in base.items():
        try:
            value.update({'gpt-answer':gpt[key]['gpt4-answer']})
            base_score=int(re.findall(r'\d+',str(value[args.model_name+'-answer']))[-1])
            gpt_score=int(re.findall(r'\d+', str(value['gpt-answer']))[-1])

            if base_score not in [0,1,2]:
                print('baseline模型答案解析错误...')
                base_score=1
            if gpt_score not in [0, 1, 2]:
                print('gpt模型答案解析错误...')
                gpt_score = 1
            base_answer.append(base_score)

            gpt_answer.append(gpt_score)
        except:
            continue
    cm = confusion_matrix(gpt_answer, base_answer)
    print(base_answer)
    cm_df=pd.DataFrame(cm,columns=['predict0','predict1','predict2'],index=['ac0','ac1','ac2'])
    print(cm_df)
    print(Counter(gpt_answer))






if __name__ == '__main__':
    # 首先进行baseline跑结果
    # generate_answer(args.input,args.output,model)
    # 与gpt4的结果进行对比，计算一致的
    get_score(args.output,'baseline_vs/gpt4.json')


"""python Eval_different_model.py --input retrieval_output/out_for_score.json --output baseline_vs/qwen14b.json --model_name qwen14b"""


