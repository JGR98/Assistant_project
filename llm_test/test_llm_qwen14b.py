import requests
import json
import re
# import openai
# from tqdm import tqdm
def qwen14b(content,temperature,top_p):
    # response = requests.post("http://36.103.203.50:8005/api/chat", data=json.dumps(
    #     {'q': content, 'repetition_penalty': 1.05, 'do_sample': True, 'top_p': top_p,
    #      'temperature': temperature}))
    # res = response.json()['response'][0]

    url = 'http://10.3.39.3:9003/generate_qwen_chat'
    data = {
        "text_input": content,
        "parameters": {
            "stream": False,
            "temperature": temperature,
            "max_tokens": 512,
        }
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    res=response.json()['text_output']
    return res

# def gpt4(content, gpt4_id, gpt4_args):
#     openai.api_type = "azure"
#     openai.api_base = "https://g4-eus2.openai.azure.com/"
#     openai.api_version = "2023-07-01-preview"
#     openai.api_key = '1b2630918e3f45fea7cf36592923b276'
#     response = openai.ChatCompletion.create(
#         engine='g4-eus-8k2',
#         messages=[{"role": "system", "content": content}],
#         temperature=0.5,
#         top_p=0.8,
#         stop=None)
#     res = response['choices'][0]['message']['content']
#     return res

def format_change(data):
    output = []
    for value in data:
        hit_answer = ''
        input_num={}
        for n,hit in enumerate(value['hits']):
            if hit['doc_id'] in input_num.keys():
                continue
            else:
                input_num[hit['doc_id']] = '研报'+str(len(input_num)+1)

            # 如果有就复用，没有就更新
            hit_answer+=input_num[hit['doc_id']] +':《' + hit['extra']['title'] + '》\t' + hit['text'] + '\n\n'
        temp = {'query': value['query'], 'hit_answer': hit_answer}
        if len(hit_answer) < 20:
            continue
        output.append(temp)
    return output


def load_json_data(path):
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    output = []
    for line in lines:
        output.append(json.loads(line))
    return output


def answer_level(query,answer):
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
             '提示：今年是2023年，去年是2022年，请注意判断时间信息和其他关键信息，尤其是标题中的实体信息。'.format(query=query, answer=answer)
    res=qwen14b(prompt,0.01,0.1)
    return res

def wenben(query,answer,display_level):
    # if display_level==2:
    #     # prompt = '你是专业资深严谨的金融分析师,请只根据资料给出专业深入正确的回答.\n\n' \
    #     #          '#####你只知道如下的信息:\n-----------------------------------\n{answer}' \
    #     #          '-----------------------------------\n#####根据上述信息回答问题:\n{question}'\
    #     #          '\n回答要求如下：\n' \
    #     #          '1. 只根据上述资料,注意*识别时间、实体、关键信息*是否匹配，然后选用正确信息，最后正确、详细、清晰、完整、高质量的回答问题。\n' \
    #     #          '2. 回答中禁止输出“根据资料”这种字眼，不要输出资料来源，不要输出资料中的“图表”等信息，直接回答问题。\n' \
    #     #          '3. 禁止反问，禁止重复问题，请完整正确的回答问题，不要混淆概念。\n' \
    #     #          '\n*提示*：今年是2023年，去年则代表着2022年；注意完整的回答问题，甄别实体和关键信息是否匹配。\n' \
    #     #          '下面请给出你对问题的回答： '.format(question=query, answer=answer)
    #
    #     prompt = '你是专业资深严谨的金融分析师,按要求给出专业深入正确的回答,\n' \
    #              '#####你只知道如下的信息(用户看不到这部分):\n-----------------------------------\n{answer}' \
    #              '-----------------------------------\n#####只根据上述信息(用户看不到这部分)回答问题:\n{question}' \
    #              '\n回答要求如下：\n' \
    #              '1. 注意*识别时间、实体、关键信息*维度上<每一条子信息>和<问题>的相关性和逻辑关系，只选用正确和相关性强的信息，最后正确、详细、清晰、全面、高质量地回答问题。\n' \
    #              '2. 禁止表达信息来源:理解*用户无法也不能看到你知道的信息*，输出“根据资料/数据/图表”这种类似的表达，会让用户疑惑信息在哪里,所以禁止输出"根据资料/数据/图表”。\n' \
    #              '3. 禁止反问，禁止重复问题！请完整正确详细的回答问题，不要混淆概念。\n' \
    #              '\n*提示*：今年是2023年，去年则代表着2022年；注意完整的回答问题，只采纳正确和强相关的信息。\n' \
    #              '下面请给出你对问题的回答： '.format(question=query, answer=answer)
    #     res=qwen14b(prompt,0.7,0.95)

    # 展示级别=1，代表不确定能不能回答问题，就会出现回答不了的情况，那么要给出辅助信息输出（见要求2）
    if display_level==1 or display_level==2:
        # prompt = '你是专业资深严谨的金融分析师,请只根据资料给出专业深入的回答.\n\n' \
        #          '#####你只知道如下的信息:\n-----------------------------------\n{answer}' \
        #          '-----------------------------------\n#####根据上述信息回答问题:\n{question}\n' \
        #          '\n回答要求如下：\n' \
        #          '1. 只根据上述资料,注意*识别时间、实体、关键信息*是否匹配，然后选用正确信息，最后正确、详细、清晰、完整、高质量的回答问题。\n' \
        #          '2. 如果根据上述资料无法准确回答问题，请在*开头*输出“我无法准确回答你的问题，但我可以为你提供如下帮助：”，然后再输出有帮助的信息。\n' \
        #          '3. 回答中禁止输出“根据资料”这种字眼，不要输出资料来源，不要输出资料中的“图表”等信息，直接回答问题。\n' \
        #          '\n*提示*：今年是2023年，去年则代表着2022年；注意完整的回答问题，甄别实体和关键信息是否匹配。\n' \
        #          '下面请给出你对问题的回答:  '.format(question=query, answer=answer)
        prompt = '你是资料问答助手，要求给出专业深入正确的回答,\n' \
                 '#####你只知道如下的资料(用户看不到这部分):\n-----------------------------------\n{answer}' \
                 '-----------------------------------\n#####只根据上述信息(用户看不到这部分)回答问题:\n{question}' \
                 '\n回答要求如下：\n' \
                 '1. 注意*识别时间、实体、关键信息*维度上<每一条子信息>和<问题>的相关性和逻辑关系，只选用正确和相关性强的信息，最后正确、详细、清晰、全面、高质量地回答问题。\n' \
                 '2. 如果根据上述资料无法准确回答问题，请在*开头*输出“我无法准确回答你的问题，但我可以为你提供如下帮助：”，然后再输出有帮助的信息。\n' \
                 '3. 禁止表达信息来源:理解*用户无法也不能看到你知道的信息*，输出“根据资料/数据/图表”这种类似的表达，会让用户疑惑信息在哪里,所以禁止输出"根据资料/数据/图表”。\n' \
                 '4. 禁止反问，禁止重复问题！请完整正确详细的回答问题，不要混淆概念。\n' \
                 '\n*提示*：今年是2023年，去年则代表着2022年；注意完整的回答问题，只采纳正确和强相关的信息。\n' \
                 '下面请给出你对问题的回答： '.format(question=query, answer=answer)
        print(prompt)
        # res = qwen14b(prompt, 0.7,0.95)
        res=''
    res=res.replace('根据上述信息，','').replace('根据资料，','').replace('根据信息，','').replace('从上述信息来看，','').replace('从上述资料来看，','').replace('从资料中可以看出，','').replace('根据提供的信息，','')
    return res

if __name__ == '__main__':
    # 加载检索数据
    data=load_json_data('out-1207.json')
    # 转换格式
    data=format_change(data)
    num=0
    output=[]
    for item in data:
        try:
            if len(item['hit_answer'])>2060:
                continue
            # 结果评级
            # res=answer_level(item['query'],item['hit_answer'])
            # print(res)
            # display_level=int(re.findall(r'\d+',str(res))[-1])
            # print('display_level:', display_level)
            # if display_level==0:
            #     continue
            # 结果评级0 1 2,  0不考虑，1和2两种不同的prompt
            # 资料问答  query是问题  hit ansser是检索单，然后displaylevel是结果评级，out是模型的输出
            out=wenben(item['query'],item['hit_answer'], 1)
            output.append({'query':item['query'],
                           'hit_answer':item['hit_answer'],
                           'display_level':1,
                           'output':out})
            print(item['query'])
            print(out)
        except:
            print('wrong')
            continue
    with open('output_test.json','w',encoding='utf-8') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)

"""
1. 不想要 根据资料 这种话语，去看还有哪些这种话术，规则替换掉
目前已经筛选出了res=res.replace('根据上述信息，','').replace('根据资料，','').replace('根据信息，','').replace('从上述信息来看，','').replace('从上述资料来看，','').replace('从资料中可以看出，','').replace('根据提供的信息，','')
2. 看大模型回答的是否正确
3. 是否有反问的，没回答的情况
4. prompt还有没有优化空间
"""