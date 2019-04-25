#coding=utf-8
import http.client
import hashlib
from urllib import parse
import random

def translate(txt_input):
    appid = '20190416000288427' #你的appid
    secretKey = 'OjhiDsnJagVUOxAoqxSz' #你的密钥
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = txt_input
    #'我他妈要红色高跟鞋\n'
    fromLang = 'zh'
    toLang = 'en'
    salt = random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode(encoding='utf-8'))
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        str_res = response.read().decode('utf-8')
        str_res = eval(str_res)
        for line in str_res['trans_result']:
            # print(line['dst'])
            return line['dst']
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

# python3模板
# /usr/bin/env python
# coding=utf8
# import http.client
# import hashlib
# import urllib.request
# import random
# import json
#
# appid = '20190416000288427' #你的appid
# secretKey = 'OjhiDsnJagVUOxAoqxSz' #你的密钥
#
# httpClient = None
# # myurl = '/api/trans/vip/translate'
# myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
# # 输入的单词
# q = 'apple'
#
# # 输入英文输出中文
# fromLang = 'en'
# toLang = 'zh'
# salt = random.randint(32768, 65536)
# sign = appid + q + str(salt) + secretKey
# m1 = hashlib.new('md5')
# m1.update(sign.encode('utf-8'))
# sign = m1.hexdigest()
# # m1 = hashlib.new('md5',sign).hexdigest()
# # m1.update(sign)
# # sign = m1.hexdigest()
# myurl = myurl + '?q=' + urllib.request.quote(
#     q) + '&from=' + fromLang + '&to=' + toLang + '&appid=' + appid + '&salt=' + str(salt) + '&sign=' + sign
# try:
#     httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
#     httpClient.request('GET', myurl)
#     # response是HTTPResponse对象
#     response = httpClient.getresponse()
#     result = response.read()
#
#     data = json.loads(result)
#     wordMean = data['trans_result'][0]['dst']
#     print(wordMean)
#
# except Exception as e:
#     print(e)
# finally:
#     if httpClient:
#         httpClient.close()
