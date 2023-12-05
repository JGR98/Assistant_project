import time
from tqdm import tqdm
import openai
import json
from Config import Config
import statistics
import argparse

Config = Config()

parser = argparse.ArgumentParser(description='evaluate llm parse')
parser.add_argument('--input', type=str, required=True, help='question type - subjective or objective')
parser.add_argument('--output', type=str, required=True, help='question type - subjective or objective')
parser.add_argument('--gpt1', type=int, required=True, help='question type - subjective or objective')
parser.add_argument('--gpt2', type=int, required=True, help='question type - subjective or objective')
args = parser.parse_args()

def gpt4(content, gpt4_id, gpt4_args):
    openai.api_type = "azure"
    openai.api_base = gpt4_args[gpt4_id]['api_base']
    openai.api_version = "2023-07-01-preview"
    openai.api_key = gpt4_args[gpt4_id]['api_key']
    response = openai.ChatCompletion.create(
        engine=gpt4_args[gpt4_id]['engine'],
        messages=[{"role": "system", "content": content}],
        temperature=0.5,
        top_p=0.8,
        stop=None)
    res = response['choices'][0]['message']['content']
    return res


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


def generate_answer(input,gpt1,gpt2,output_path):
    output = []
    retrieval_answer = load_json_data(input)
    data = format_change(retrieval_answer)
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

            answer1 = gpt4(prompt, gpt1, Config.gpt4_version)
            answer2 = gpt4(prompt, gpt2, Config.gpt4_version)
            time.sleep(10)
            value.update({'input': prompt, 'gpt4-answer1': answer1, 'gpt4-answer2': answer2})
            output.append(value)
        except:
            print('wrong')
            continue
    save_json_data(path=output_path, data=output)

# 改写成一个一个处理的

def check_stable_answer(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    nums=0
    output=[]
    for value in data:
        answer1 = json.loads(value['gpt4-answer1'])
        answer2 = json.loads(value['gpt4-answer2'])
        value.update({'score1': answer1['score'],'score2':answer2['score']})
        if answer1['score']==answer2['score']:
            value.update({'gpt4-answer':answer1})
        else:
            value.update({'diff': 1})
            nums+=1
        output.append(value)
    print(nums)
    return output


def regenerate_answer(data,output_path):
    # 跑，然后还要给出最终的正确结果
    output=[]
    for value in tqdm(data):
        if 'diff' in value.keys():
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
            answer3 = gpt4(prompt, 6, Config.gpt4_version)
            score_list=[int(value['score1']),int(value['score2']),int(json.loads(str(answer3))['score'])]
            mode= statistics.mode(score_list)
            ind=score_list.index(mode)
            value.update({'gpt4-answer3': answer3,'gpt4-answer':value['gpt4-answer'+str(ind+1)]})
        output.append(value)
    save_json_data(path=output_path, data=output)
    # 然后要保存
    return None


if __name__ == '__main__':
    # 首先进行gpt4跑结果
    generate_answer(args.input,args.gpt1,args.gpt2,args.output)
    # 然后检查结果是否一致
    data=check_stable_answer(args.output)
    # 不一致的重跑，然后以多数的作为最终结果
    regenerate_answer(data,args.output)

"""python Gpt4_get_level_answer.py --input retrieval_output/out-第一版.json --output train_data/output_1.json --gpt1 9 --gpt2 10"""