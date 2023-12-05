# _*_coding:utf-8_*_
import requests
import json
# 在10.3.39.3服务器可以调用
class qwen7b():
    def __init__(self):
        self.name='qwen7b'
    def run(self,content):
        url = "http://10.3.39.3:8806/stuck_work/api/qwapi/"
        payload = {
            "text": content,
            "temperature": 0.01,
            "top_k": 10,
            "top_p": 0.1
        }
        headers = {
            "content-type": "application/json"
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        content = response.content.decode('utf8')
        return content

class qwen14b():
    def __init__(self):
        self.name='qwen14b'
    def run(self,content):
        response = requests.post("http://36.103.203.50:8005/api/chat", data=json.dumps(
            {'q': content, 'repetition_penalty': 1.05, 'do_sample': True, 'top_p': 0.7,
             'temperature': 0.01}))
        res = response.json()['response'][0]
        return res

