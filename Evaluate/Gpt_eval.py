import json
import time

from Config import Config
import openai
import re

# 对检索的结果进行一条一条的评价，输入然后评价，每一条给一个得分，只看top5的准确率
class Eval_process():
    def __init__(self, config):
        self.config = config

    def gpt4(self, content, gpt4_id, gpt4_args):
        openai.api_type = "azure"
        openai.api_base = gpt4_args[gpt4_id]['api_base']
        openai.api_version = "2023-07-01-preview"
        openai.api_key = gpt4_args[gpt4_id]['api_key']
        response = openai.ChatCompletion.create(
            engine=gpt4_args[gpt4_id]['engine'],
            messages=[{"role": "system", "content": content}],
            temperature=0.01,
            top_p=0.05,
            stop=None)
        res = response['choices'][0]['message']['content']
        return res

    def eval_retrieval(self, ques, ans, gpt4_id):
        gpt4_args = self.config.gpt4_version
        prompt = '你是一个检索评估专家，你需要对用户输入的“检索问题”和”检索答案“进行相关性判断(打标签):非常相关，一般相关，不相关。\n' \
                 '###判断标准如下:\n' \
                 '·非常相关-[两者之间有直接的关联，查询结果与文本内容高度一致],\n' \
                 '·一般相关:[两者之间有一定的关联，查询结果与文本内容有一定的相似性],\n' \
                 '·不相关:[两者之间没有关联，查询结果与文本内容没有任何相似性]\n' \
                 '提示：判断是请关注语境是否相关(主语,时间,地点,人物是否能对齐)；如果检索答案中有问题的答案则算作非常相关\n' \
                 '输出格式为json；{{"reason":<评价理由>, "level":<非常相关 or 一般相关 or 不相关> }}\n\n' \
                 '-------------------------------------------------------\n' \
                 '检索问题：\t{question}\n\n' \
                 '检索答案：\t{answer}\n' \
                 '-------------------------------------------------------\n' \
                 '\n请按照json格式给出评价，并注意评分标准和提示内容：'.format(question=ques, answer=ans)
        answer = self.gpt4(prompt, gpt4_id, gpt4_args)
        # 直接给出一个格式化的得分
        level='a'+re.findall(r'(非常相关|一般相关|不相关)', str(answer))[-1]
        score = 2 if '非常相关' in level else 1 if '一般相关' in level else 0 if '不相关' in level else None
        try:
            res = {"reason": json.loads(answer)['reason'], "level": json.loads(answer)['level'], "score": score}
        except:
            res = {"reason": answer, "level": re.findall(r'(非常相关|一般相关|不相关)',str(answer))[-1], "score": score}

        return res

    # 对 展示级别去跑baseline_eval的代码就行

    # 大模型评价
    def eval_qa(self, question,hit_answer,answer):
        prompt = '你是一个评估专家，请判断下面根据资料的问答，是否正确的回答了问题。\n' \
                 '从两方面判断：\n' \
                 '1. 准确性accuracy：根据资料判断是否回答正确了，没有使用资料中的错误信息，没有输出错误信息（输出无法回答也算作没有错误信息）（正确得1分，错误得0分）\n' \
                 '2. 响应度responsiveness：是否真正回答了问题，解决了问题，而不是回复了无用信息和没有作答（回答了得1分，没回答得0分）。\n' \
                 '\n--------------------------------------------------\n' \
                 '####问题{query}\n\n' \
                 '####资料：\n{text}\n\n' \
                 '####答案：\n{answer}\n' \
                 '\n--------------------------------------------------\n' \
                 '请以json的格式输出你的评价结果{{"reason":<评价理由>, "accuracy":<0 or 1> ,"responsiveness":<0 or 1>}}'.format(
            query=question,text=hit_answer,answer=answer
        )
        try:
            answer = self.gpt4(prompt, 7,self.config.gpt4_version)
        except:
            answer='{"reason":<gpt执行失败>, "accuracy":0 ,"responsiveness":0}'
        # 直接给出一个格式化的得分
        try:
            res = {"hit_answer":hit_answer,"reason":json.loads(answer)['reason'],"accuracy": json.loads(answer)['accuracy'], "responsiveness": json.loads(answer)['responsiveness']}
        except:
            res = {"hit_answer":hit_answer,"reason": answer.split('reason')[-1], "accuracy": re.findall(r'\d+',answer)[-2],
                   "responsiveness": re.findall(r'\d+',answer)[-1]}
            print(answer)
        time.sleep(10)
        return res
