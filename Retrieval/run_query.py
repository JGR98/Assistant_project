# _*_coding:utf-8_*_
import argparse
import requests
import json

def run(url, ifile, ofile):
    fout = open(ofile,'w')
    for q in open(ifile,'r').readlines():
        if q.strip():
            id, query = q.split('\t')
            print(query)
            body = {'botid': 1, 'ask_content': query, "skip_answerable": 1}
            res = requests.post(url, json=body)
            if res.ok:
                res_dict = res.json()
                res_dict['query'] = query
                fout.write(json.dumps(res_dict, ensure_ascii=False))
                fout.write('\n')

    fout.close()

parser = argparse.ArgumentParser(description='run retrieval')
parser.add_argument('--ifile',type=str,help='input file')
parser.add_argument('--ofile',type=str, help='output file')
parser.add_argument('--url', type=str, help='retrieval url')
args = parser.parse_args()
run(args.url, args.ifile, args.ofile)
"""

python run_query.py --ifile 1019.txt --ofile out.json --url "http://10.5.113.131:9001/invoke?json=true"
"""