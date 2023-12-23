import time

import requests

# 定义 API 地址


import requests
import json

def send_post_request(question_list):
    url = "http://10.5.113.131:8047/aihelper/search_list/"
    headers = {'Content-Type': 'application/json'}
    data = {"question_list": question_list}
    print(data)
    response = requests.post(url, headers=headers, json=data)
    return response.text

def main():

    with open('1220-001.txt','r',encoding='utf-8') as f:
        # data=json.load(f)
        # f.close()
        data=f.readlines()

    for key in data:
        try:
            question_list = [key.strip()]
            response_text = send_post_request(question_list)
            print(key)
            with open('1220-menghuan.json', 'a', encoding='utf-8') as f:
                json.dump(response_text, f, ensure_ascii=False, indent=2)
                f.close()
            print('sus')
            time.sleep(90)
        except:
            print('wrong')
            continue
    # 将响应写入文件

if __name__ == "__main__":
    main()

