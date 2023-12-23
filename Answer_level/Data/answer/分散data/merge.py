import json
import re

with open('output_1211.json', "r", encoding='utf-8') as f:
    a = json.load(f)
    f.close()
with open('output_1211-1.json', "r", encoding='utf-8') as f:
    b = json.load(f)
    f.close()
with open('output_1221-2.json', "r", encoding='utf-8') as f:
    c = json.load(f)
    f.close()
with open('output_1211-3.json', "r", encoding='utf-8') as f:
    d = json.load(f)
    f.close()
with open('output_1221-4.json', "r", encoding='utf-8') as f:
    e = json.load(f)
    f.close()
with open('output_1211-5.json', "r", encoding='utf-8') as f:
    t = json.load(f)
    f.close()

data=a+b+c+d+e+t
data0=[]
data1=[]
data2=[]
output=[]
safe=[]
i=0
m=0
n=0
j=0
for item in data:
    score=int(re.findall('\d+',str(item['gpt4-answer']))[-1])
    if score==0:
        data0.append(item)
        output.append(item)
        m+=1

    elif score==1:
        data1.append(item)
        if i%4==0:
            output.append(item)
            n += 1
        else:
            safe.append(item)

    else:
        data2.append(item)
        i+=1
        if i%5==0:
            output.append(item)
            j += 1
        else:
            safe.append(item)

# print(len(data0))
# print(len(data1))
# print(len(data2))
# print(len(output))
# print(len(safe))
# print(m)
# print(n)
# print(j)
with open('output_1211_aaa.json', "r", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    f.close()
# with open('stay.json', "w", encoding='utf-8') as f:
#     json.dump(safe, f, ensure_ascii=False, indent=2)
#     f.close()


