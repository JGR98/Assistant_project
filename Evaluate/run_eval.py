import time
import warnings
import numpy as np
import pandas as pd
from Gpt_eval import Eval_process
import json
from tqdm import tqdm
from Config import Config
import argparse
import re
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
Config = Config()
eval_process = Eval_process(config=Config)


def load_line_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    data = []
    for line in lines:
        data.append(json.loads(line))
    print('success load file : ', path)
    return data


def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    print('success load file : ', path)
    return data
def load_daguan_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    # print(len(re.findall('}\n{',data)))
    output=[]
    for i,item in enumerate(data.split('}\n{')):
        if i ==0:
            item=item+'}'
        elif i==len(data.split('}\n{'))-1:
            item='{'+item
        else:
            item='{'+item+'}'
        output.append(json.loads(item))
    print('success load file : ', path)
    return output


def save_json_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.close()
    print('success save file : ', path)
    return None


def retrieval_answer_format_change(hits):
    hit_answer_list = []
    hit_answer = ''
    for hit in hits:
        hit_answer += '《' + hit['extra']['title'] + '》\t' + hit['text'] + '\n\n'
        hit_answer_list.append('《' + hit['extra']['title'] + '》\t' + hit['text'])

    return hit_answer, hit_answer_list


def web_answer_format_change(hits):
    hit_answer_list = []
    hit_answer = ''
    for hit in hits:
        hit_answer += '《' + hit['title'] + '》\t' + hit['content'] + '\n\n'
        hit_answer_list.append('《' + hit['title'] + '》\t' + hit['content'])
    return hit_answer, hit_answer_list


def whole_first_process(path):
    # 需要先判断这个数据是不是已经被处理过了的结果

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = json.load(f)
    except:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read()
        end = len(lines.split('}""{'))
        output = []
        for i, line in enumerate(lines.split('}""{')):
            if i == 0:
                line = line + '}"'
            elif i == end - 1:
                line = '"{' + line
            else:
                line = '"{' + line + '}"'
            try:
                output.append(json.loads(line))
            except:
                print('wrong',i)
        save_json_data(output,path)

    return None

# 解析检索答案
def parse_retrieval_data(retreival_data, model, save_path):
    # 解析检索的答案，变成处理好的query拼接+querylist
    output = []
    keys = []
    for value in retreival_data:
        hit_answer, hit_answer_list = retrieval_answer_format_change(value['hits'])
        temp = {
            value['query'].strip(): {'query': value['query'], 'hit_answer_' + model: hit_answer,
                                     'hit_answer_list_' + model: hit_answer_list}}
        if len(hit_answer) < 20:
            continue
        keys.append(value['query'].strip())
        output.append(temp)
    print('**retrieval data length is :',len(output))
    if model == 'assis':
        pd.DataFrame(keys).to_csv('Data/retrieval/retrieval_query/retrieval_query.txt', sep='\t')
    save_json_data(output, save_path)
    return output


def parse_retrieval_data_daguan(retreival_data, model, save_path):
    # 解析检索的答案，变成处理好的query拼接+querylist
    output = []
    keys = []
    for value in retreival_data:

        hits=value['data']['items'][0]['data']['items']
        hit_answer_list = []
        hit_answer = ''
        for hit in hits[:7]:
            try:
                if len(hit_answer_list)>=5:
                    break
                hit_answer += '《' + hit['data']['title'] + '》\t' + hit['data']['content'] + '\n\n'
                hit_answer_list.append('《' + hit['data']['title'] + '》\t' + hit['data']['content'])
            except:
                continue
        temp = {
            value['query'].strip(): {'query': value['query'], 'hit_answer_' + model: hit_answer,
                                     'hit_answer_list_' + model: hit_answer_list}}

        if len(hit_answer) < 20:
            continue
        keys.append(value['query'].strip())
        output.append(temp)

    save_json_data(output, save_path)
    return output


# 解析全部的答案
def parse_llm_data(data_path, model, save_path):
    # 解析检索的答案，变成处理好的query拼接+querylist
    whole_first_process(data_path)
    data = load_json_data(data_path)
    output = []
    print(len(data))
    for value in data:
        value=json.loads(value)
        try:
            temp = {}
            for item in value['data'][0]['process']:
                if item['search_name'] =='研报搜索':
                    temp['hit_report']=retrieval_answer_format_change(item['response_json']['hits'])[0]
                    temp['display_level_report']= item['response_json']['display_level']
                elif item['search_name'] =='BI搜索':
                    temp['hit_bi'] = item['response_json']
                elif item['search_name'] =='财务搜索':
                    temp['hit_finan'] = retrieval_answer_format_change(item['response_json']['hits'])[0]
                    temp['display_level_finan'] = item['response_json']['display_level']
                elif item['search_name'] == '人力搜索':
                    temp['hit_man'] = retrieval_answer_format_change(item['response_json']['hits'])[0]
                    temp['display_level_man'] = item['response_json']['display_level']
                elif item['search_name'] =='外搜提取段落':
                    temp['hit_web'] = web_answer_format_change(item['response_json']['hits'])[0]
                elif item['search_name'] =='意图识别':
                    temp['intent'] = item['response_json']['response'].strip().split()
            # 对于没有的search_name
            temp['llm_out']=value['data'][0]['data']['answer']
            temp['answer_from']=value['data'][0]['data']['from']
            temp['message']=value['data'][0]['data']['message']
            temp['query']=value['data'][0]['question']
            output.append(temp)
        except Exception as e:
            print('(***error occur:\n', e)
            continue
    save_json_data(output, save_path)
    return output


# 计算分数，每条都添加一个score的字段
def calculate_score(model, path, task):
    data = load_json_data(path)

    if task == 'retrieval':
        df = pd.DataFrame(
            columns=['id', 'query', 'type', 'hit_answers', 'llm_out','score', 'score_rate-0', 'score_rate-1', 'score_rate-2'])
        for value in data:
            key = list(value.keys())[0]
            df = pd.concat([df, pd.DataFrame(
                {'id': 0, 'query': key, 'type': value[key]['type'], 'hit_answers': value[key]['hit_answer_' + model],
                 'llm_out':str(value[key]['llm_out']),
                 'score': str(value[key]['score']), 'score_rate-0': value[key]['score_rate']['0'],
                 'score_rate-1': value[key]['score_rate']['1'], 'score_rate-2': value[key]['score_rate']['2']},
                index=[0])],
                           ignore_index=True)
        # 计算平均得分
        df = pd.concat([df, pd.DataFrame({'score_rate-0': df['score_rate-0'].mean(),
                                          'score_rate-1': df['score_rate-1'].mean(),
                                          'score_rate-2': df['score_rate-2'].mean()}, index=['avg'])],
                       ignore_index=False)
        df = pd.concat([df, df.groupby('type')[['score_rate-0', 'score_rate-1', 'score_rate-2']].mean()],
                       ignore_index=False)
        # 保存结果
        with pd.ExcelWriter('Data/result/retrieval_eval_score.xlsx', engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            # 将DataFrame追加到现有的Excel文件的指定sheet中
            df.to_excel(writer, sheet_name=model)
        print('**** success save retrieval eval score!!!')
    elif task == 'llm':
        # 构建成一个dataframe,，保存的项目更多了
        df = pd.DataFrame(
            columns=['id', 'query', 'type','from','message', 'hit_answer','llm_out',
                      'score_acc', 'score_res','gpt4-eval'])

        for value in data:
            if len(str(value['llm_out']))<5:
                continue
            df = pd.concat([df, pd.DataFrame({'id': value['query'], 'query': value['query'], 'type': value['type'],
                                              'from':str(value['answer_from']),
                                              'message': str(value['message']),
                                              'llm_out':str(value['llm_out']),
                                              'hit_answers':str(value['hit_answer']),
                                              'score_acc': int(value['score_acc']),
                                              'score_res': int(value['score_res'])
                                             }, index=[0])],
                           ignore_index=True)

        # 计算平均得分
        df = pd.concat([df, pd.DataFrame({'score_acc': df['score_acc'].mean(), 'score_res': df['score_res'].mean()},
                                         index=['avg'])],
                       ignore_index=False)
        df = pd.concat([df, df.groupby('type')[['score_acc', 'score_res']].mean()],
                       ignore_index=False)
        df = pd.concat([df, df.groupby('message')[['score_acc', 'score_res']].mean()],
                       ignore_index=False)
        df = pd.concat([df, df.groupby('from')[['score_acc', 'score_res']].mean()],
                       ignore_index=False)
        df = pd.concat([df, df.groupby(['from','type'])[['score_acc', 'score_res']].mean()],
                       ignore_index=False)
        # 保存结果
        with pd.ExcelWriter('Data/result/llm_eval_score2.xlsx', engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            # 将DataFrame追加到现有的Excel文件的指定sheet中
            df.to_excel(writer, sheet_name=model)

        print('**** success save llm eval score!!!')
    return None


# 评估模型的最终输出
def whole_eval(model, path_whole_parse, path_query_type, path_whole_eval,config):
    # 加载数据
    data = load_json_data(path_whole_parse)
    data_type = load_json_data(path_query_type)
    llm_eval_output = []
    # 评估大模型的输出
    for value in tqdm(data):
        # 判断最终的意图对应的是哪个hit，然后再进行评估，并进行分类
        if '请求超时' in value['answer_from']:
            value.update(
                {'gpt4_eval': "", 'hit_answer': '', 'score_acc': 0,
                 'score_res': 0,
                 'type': data_type[value['query']]['type'], 'id': data_type[value['query']]['id']})
        elif '摘要' in value['answer_from']:
            value.update(
                {'gpt4_eval': "",'hit_answer':value[config.mapping_intent[value['answer_from']]], 'score_acc': 1, 'score_res': 1,
                 'type': data_type[value['query']]['type'], 'id': data_type[value['query']]['id']})
        elif '闲聊' in value['answer_from'] :
            value.update(
                {'gpt4_eval': "", 'hit_answer': "", 'score_acc': 1,
                 'score_res': 1,
                 'type': data_type[value['query']]['type'], 'id': data_type[value['query']]['id']})
        else:
            answer = eval_process.eval_qa(value['query'], value[config.mapping_intent[value['answer_from']]], value['llm_out'])
            value.update(
                {'gpt4_eval': answer, 'hit_answer': answer['hit_answer'],'score_acc': answer['accuracy'],
                 'score_res': answer['responsiveness'], 'type': data_type[value['query']]['type'], 'id': data_type[value['query']]['id']})
        llm_eval_output.append(value)

    save_json_data(llm_eval_output, path_whole_eval)
    print('**** End LLM_OUT Eval')
    return None


def retrieval_eval(model, path_retrieval_answer_org, path_query_type, path_save_retrieval_parse,
                   path_save_retrieval_eval):
    # 加载数据
    if model=='assis':
        data_org = load_line_data(path_retrieval_answer_org)
        # 解析数据
        data = parse_retrieval_data(retreival_data=data_org, model=model,
                                    save_path=path_save_retrieval_parse)
    elif model=='daguan':
        data_org = load_line_data(path_retrieval_answer_org)
        # 解析数据
        data = parse_retrieval_data_daguan(retreival_data=data_org, model=model,
                                    save_path=path_save_retrieval_parse)

    # print(data_org.keys())
    print(len(data_org))


    data_type = load_json_data(path_query_type)
    # 去跑检索的评分
    retrieval_eval_output = []
    gpt4_ids = [1, 4, 5, 6, 7]
    for num, value in enumerate(tqdm(data)):
        gpt4_eval = []
        score = []
        key = list(value.keys())[0]
        for g_id, hit_ans in enumerate(value[key]['hit_answer_list_' + model]):
            try:
                # gpt4评价检索的结果，每次调用不同的gpt4接口
                answer = eval_process.eval_retrieval(ques=key, ans=hit_ans, gpt4_id=gpt4_ids[g_id])
                gpt4_eval.append(answer)
                score.append(answer['score'])
            except:
                print('wrong for query:',key)
                score.append(np.nan)
                continue
        score_rate = {0: score.count(0) / len(score), 1: score.count(1) / len(score), 2: score.count(2) / len(score)}
        value[key].update(
            {'gpt4_eval': gpt4_eval, 'score': score, 'score_rate': score_rate, 'type': data_type[key]['type'],
             'id': data_type[key]['id']})
        retrieval_eval_output.append(value)
        time.sleep(12)

    # 结果要保存
    save_json_data(retrieval_eval_output, path_save_retrieval_eval)
    print('**** End Retrieval Eval')
    return None


def displaylevel_eval(path_whole_eval):
    # 换一种思路，把gpt4的答案拼到原文中
    data = load_json_data(path_whole_eval)
    base_answer=[]
    gpt_answer=[]
    for value in data:
        try:
            # base_score = value['display_level_report']
            base_score = int(re.findall(r'\d+', str(value['qwen18b_lora-answer']))[-1])
            gpt_score = int(re.findall(r'\d+', str(value['gpt4-answer']))[-1])
            if base_score not in [0, 1, 2]:
                base_score = 3
            if gpt_score not in [0, 1, 2]:
                gpt_score = 3
            base_answer.append(base_score)
            gpt_answer.append(gpt_score)
        except:
            print(1)
            continue
    cm = confusion_matrix(gpt_answer, base_answer)
    print(cm)
    cm_df = pd.DataFrame(cm, columns=['predict0', 'predict1', 'predict2','other'],
                         index=['ac0', 'ac1', 'ac2','other'])
    print(cm_df)

def chat_intent(data):
    # 只判断闲聊的意图是否准确了
    for item in data:
        if '闲聊' in item['intent']:''
    # bi的也要判断意图是否准确
    # 有的应该走外搜却没走的？  感觉意图识别现在没那么重要



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse')
    # parser.add_argument('--type', type=str, required=True, help='question type - subjective or objective')
    parser.add_argument('--model', type=str, help='评估哪个模型')
    parser.add_argument('--task', type=str, help='评估什么环节')
    args = parser.parse_args()
    model = args.model
    task = args.task
    path_query_type = 'Data/test_query_e2e.json'
    if task == 'retrieval':
        path_retrieval_answer_org = 'Data/retrieval/retrieval_answer/retrieval_out_'+model+'.json'
        path_retrieval_parse = 'Data/retrieval/retrieval_answer_after_parse/' + model + '_retrieval_answer.json'
        path_save_retrieval_eval = 'Data/retrieval/retrieval_eval/' + model + '_retrieval_eval.json'

        # 检索评估
        # retrieval_eval(model,path_retrieval_answer_org=path_retrieval_answer_org,
        #                               path_query_type=path_query_type,path_save_retrieval_parse=path_retrieval_parse,
        #                               path_save_retrieval_eval=path_save_retrieval_eval)

        # 计算得分
        score = calculate_score(model, path_save_retrieval_eval, task)
    # 评价大模型的，要给出两块得分之后，两块都要看，同样搞成dataframe的样式

    if task == 'llm':
        path_whole_answer_org = 'Data/whole_process/whole_answer/whole_output.json'
        path_whole_parse = 'Data/whole_process/whole_after_parse/' + model + '_whole_answer.json'
        path_whole_eval = 'Data/whole_process/whole_eval/' + model + '_whole_eval.json'
        #
        parse_llm_data(path_whole_answer_org, model=model, save_path=path_whole_parse)

        # whole_eval(model=model, path_whole_parse=path_whole_parse, path_whole_eval=path_whole_eval,
        #            path_query_type=path_query_type,config=Config)
        score = calculate_score(model, path_whole_eval, task)
    if task =='display_level':
        # path_whole_eval = 'Data/whole_process/whole_eval/' + model + '_whole_eval_with_level.json'
        path_whole_eval = 'Data/whole_process/whole_eval/qwen18b_lora_1222_250.json'
        displaylevel_eval(path_whole_eval=path_whole_eval)



