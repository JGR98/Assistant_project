# 需要构建多样性的话术，借助一下gpt4
import openai
import json
import random

def gpt4(content):
    openai.api_type = "azure"
    openai.api_base = "https://g4-eus2.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = '1b2630918e3f45fea7cf36592923b276'
    response = openai.ChatCompletion.create(
        engine="g4-eus-8k2",
        messages=[{"role": "system", "content": content}],
        temperature=0.95,
        top_p=0.95,
        stop=None)
    res = response['choices'][0]['message']['content']
    return res

def generate_diff_prompt():
    prompt='我现在想要构建lora微调训练数据集，所以需要构建多样化的prompt，请帮我把下面的话术给改写成非常多样化的,请尽量多样化的变换，可以删掉无关语句，话术如下：\n\n'\
            '您是一个检索系统评估专家。您的任务是评估检索系统给出的信息，确定它们是否能够回答特定的问题。' \
         '请注意，回答可能来自单条信息，或者通过多条信息的关联推理得出。请仔细分析每条信息，并给出总体评分。\n\n' \
         '**评分标准**：\n ·如果有信息*能够回答*问题，或者通过*推理得出答案*，给予<2分>;\n ' \
         '·如果根据信息*没有把握回答*问题，但是能*提供一些帮助*，给予<1分>;\n ·如果信息*不能回答问题*，也*没有提供任何帮助*，给予<0分>\n\n' \
         '请在给出评分时，提供您的推理过程和判断依据。\n\n' \
         '### 问题:\n------------------------------------------------------------\n{query}' \
         '------------------------------------------------------------' \
         '\n\n### 检索答案:\n------------------------------------------------------------\n{answer}' \
         '------------------------------------------------------------\n\n' \
         '请给出理由和评分，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n' \
         '提示：今年是2023年，去年是2022年，请注意判断时间信息和其他关键信息。'
    return gpt4(prompt)

# gpt4生产出来的10个话术
instruction={
    1:'你是一名搜索系统评估大师。你需要评估出自搜索系统的信息，确认它们能否满足对特定问题的答案需求。请留意，答案可能来自单独的一条信息，或者是多个信息共同推理得到的。详细阅读每一条信息，然后提供总体评价。\n\n**评价标准**：\n ·如果有信息*能解答*问题，或通过*推理获得答案*，得<2分>；\n ·如果信息*不能确定回答*问题，但是能*有所帮助*，得<1分>；\n ·如果信息*不能解答问题*，且*没有任何帮助*，得<0分>\n\n在评价时，请给出你的推理过程和评价理由。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 搜索结果:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请提供评价理由和评分，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n注：现在是2023年，去年是2022年，请注意判断时间信息和其他重要信息。'
    ,2:'作为一名搜索系统评估的专家，你的职责是评估由搜索系统提供的信息，判定他们能否解答特定的问题。注意，答案可能来自一条信息，或者由多条信息的相关推理得出。你需要仔细分析每条信息，并给出总体评分。\n\n**评分依据**：\n ·如果信息*可以回答*问题，或者可以通过*推理得出答案*，评<2分>；\n ·如果根据信息*无法确定回答*问题，但是有*一定帮助*，评<1分>；\n ·如果信息*无法回答*问题，也*没有提供帮助*，评<0分>\n\n当你给出评分时，请提供你的推理过程和评分理由。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 检索回答:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请提供评分和理由，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n备注：今年是2023年，去年是2022年，请注意关于时间和其他关键信息的判断。'
    ,3:'作为一名查询系统评估专家，你的任务是评估查询系统提供的信息，判断它们能否回答特定的问题。请记住，答案可能来自一条信息，或者可以通过多条信息关联推理得到。你需要细心研究每一条信息，并给出总体评价。\n\n**评分规则**：\n ·如果有信息*可以回答*问题，或者可以通过*推理得出答案*，打<2分>；\n ·如果根据信息*不能确定回答*问题，但是有*一定帮助*，打<1分>；\n ·如果信息*不能回答*问题，也*未能提供任何帮助*，打<0分>\n\n在打分时，请给出你的推理过程和评分依据。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 检索的答案:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请给出你的评分和原因，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n温馨提示：现在是2023年，去年是2022年，请在评估时注意判断时间信息和其他关键信息。'
    ,4:'您作为一个专业的信息检索评估师，需要对检索系统提供的信息进行全面审视，以判断这些信息能否满足问题的需求。请记住，有时答案可能只由一条信息提供，而有时则需要通过多条信息的关联推理来得出。在评估每一条信息时，您需要进行深入的分析，并为其总体效果给出评价。\n\n**评价指南**：\n ·对于能够直接*回答问题*或通过*推理得出答案*的信息，您应给出<2分>的评分；\n ·对于虽然*不能确切回答问题*但能够*提供部分帮助*的信息，您应给出<1分>的评分；\n ·对于*无法回答问题*且*未提供任何有效帮助*的信息，您应给出<0分>的评分。\n\n在评价时，请描述您的推理步骤和评分依据。\n\n### 待解答问题：\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 提取的答案：\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请描述评价理由并给出评分，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n提示：现在是2023年，上一年度为2022年，因此在评价信息时请留意时间线及其它关键信息。'
    ,5:'作为一名信息检索系统的评估专家，您需要评估检索系统提供的信息并确定它们是否能够满足指定问题的需求。答案可能直接来源于一条信息，也可能通过多条信息的关联推理而来。请您深入分析每一条信息，并为其给出一个总体评分。\n\n**评分规则**：\n ·如果信息*能够满足*问题的需求，或者答案能够通过*推理获得*，那么请给出<2分>;\n ·如果信息*不能确定*是否能满足问题的需求，但是*能提供一些帮助*，那么请给出<1分>;\n ·如果信息*不能满足*问题的需求，且*未提供任何帮助*，那么请给出<0分>\n\n在给出评分时，同时附上您的推理过程和判断依据。\n\n### 需要解答的问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 搜索到的答案:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请提供评分的理由和数字，格式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n注意：目前是2023年，去年是2022年，请在评估信息时注意时间线和其他重要信息。'
    ,6:'作为一个评估查询系统的专家，您的职责是对系统返回的信息进行评析，确保它们是否能回答特定问题。情况可能是这样，答案可能仅来自一个信息源，也可能通过多个信息源的推理得出。请您对每一条信息进行深入分析，并最后给出一个总评。\n\n**打分指南**：\n ·若信息*直接回答*了问题，或者可以*通过推理得出答案*，请打<2分>;\n ·若信息虽*不能确定回答*问题，但是能给出*部分帮助*，请打<1分>;\n ·若信息*并未回答问题*，且*完全无帮助*，请打<0分>\n\n在打分的同时，需提供您的推理和判断依据。\n\n### 查询:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 查询结果:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请以json的格式提供理由和打分，例如：{{"reason":<>,"score":<2> or <1> or <0>}}\n提示：今年是2023年，所以请注意判断时间和其他关键信息。'
    ,7:'作为一个专业的信息检索系统评价师，你的职责就是仔细考察由搜索系统得出的答案，然后决定这些答案是否能完美地解决特定的疑问。记住，答案可能由一条信息给出，也可能由多条信息联系推理得出。你需要细致的阅读每条信息，然后根据它们的质量给出总评。\n\n**评估准则**：\n ·如果存在信息*能直接解答*问题，或者通过*联系推理得出答案*，请给予<2分>;\n ·如果基于信息*无法准确回答*问题，但是可以*提供部分指导*，请给予<1分>;\n ·如果信息*无法解答问题*，且*无法提供任何帮助*，请给予<0分>\n\n在打分时，请附上你的推理流程和决策依据。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 检索结果:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请以json格式给出推理依据和得分，格式为{{"reason":<>,"score":<2> or <1> or <0>}}\n注：今年是2023年，去年是2022年，请注意在评价时考虑时间信息和其他关键信息。'
    ,8:'你的任务是审查搜寻系统所提供的答案，并判定他们是否能针对特定问题提供有用的回答。要注意，解答可能来自单一的信息，或是由多条信息相互关联推断出来的。请你对每个信息作出详尽的分析，并根据他们的质量给出一个总评。\n\n**评估指标**：\n ·若信息能*直接解答*问题，或是经过*推断得出答案*，请打<2分>;\n ·若信息*未能确切回答*问题，但是有*部分参考价值*，请打<1分>;\n ·若信息*无法解答问题*，且*毫无参考价值*，请打<0分>\n\n在评分时，请提供你的推理过程和判断的依据。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 搜索结果:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请提供评分和理由，形式为json{{"reason":<>,"score":<2> or <1> or <0>}}\n温馨提示：今年是2023年，去年是2022年，请注意审查时间信息和其他重要信息。'
    ,9:'你的职责是作为检索系统的评估专家。你需要评价检索系统提供的信息，确定这些信息是否能够回答特定的问题。请注意，回答可能源自单条信息，或是由多条信息联合推理得出。请仔细研究每条信息，并根据其质量给出总评。\n\n**评估标准**：\n ·如果有信息*可以回答*问题，或者通过*推理得出答案*，请打<2分>;\n ·如果根据信息*无法确定回答*问题，但是可以*提供一些帮助*，请打<1分>;\n ·如果信息*无法回答问题*，也*没有提供任何帮助*，请打<0分>\n\n打分时，请提供你的推理过程和判断依据。\n\n### 问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 检索回答:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请以json格式提供理由和评分，格式为{{"reason":<>,"score":<2> or <1> or <0>}}\n提示：现在是2023年，去年是2022年，请在评价时注意时间和其他关键信息。'
    ,10:'作为搜索引擎评价专家，您将被要求对搜索引擎所返回的信息进行评估，来判断这些信息是否能有效地解答一些特定的问题。这些答案有可能来自一条信息，也可能是由多条信息关联推理得出的。请您仔细地分析每一条信息，然后给出您的总体评分。\n\n**评分方式如下**：\n·如果您发现某条信息*可以解答*这个问题，或者可以通过*推理*来得到答案，那么这条信息将被赋予<2分>;\n·如果一条信息不能确切地解答问题，但*可以提供一些帮助*，那么这条信息将被赋予<1分>;\n·如果一条信息*对于解答问题没有任何帮助*，那么这条信息将被赋予<0分>\n\n在评分的过程中，我们希望您可以提供您的推理过程和判断依据。\n\n### 待解答的问题:\n------------------------------------------------------------\n{query}------------------------------------------------------------\n\n### 搜索引擎返回的答案:\n------------------------------------------------------------\n{answer}------------------------------------------------------------\n\n请给出您的推理过程和评分，并以json格式呈现{{"reason":<>,"score":<2> or <1> or <0>}}\n注意：请您记住当前是2023年，上一年是2022年，因此请您在评判的过程中注意时间信息和其他重要信息。'
}

# 随机数，随机拼接prompt上去


with open(r'train_query.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

output = []
for n, value in enumerate(data):
    num = random.randint(1, 10)
    input_conv = instruction[num].format(query=value['llm_out']['query'], answer=value['llm_out']['hit_answer'])

    temp = {
        "id": str(n),
        "conversations": [
            {
                "from": "user",
                "value": input_conv
            },
            {
                "from": "assistant",
                "value": str(value['llm_out']['gpt4-answer'])
            }
        ]
    }
    output.append(temp)
with open(r'final_for_train/lora_data_v1_1205.json', 'w', encoding='utf-8') as f:
    json.dump(output,f,ensure_ascii=False,indent=2)