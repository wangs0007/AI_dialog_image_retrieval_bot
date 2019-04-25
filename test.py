# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sim_user
import math
import ranker
import json
import random
import time
import sys
from model import NetSynUser
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from translate import *
import re
import copy
from flask import Flask, request, jsonify
app = Flask(__name__)

class Model_Loader(object):
    __instance = None
    __first_init = True

    def __init__(self):
        if self.__first_init:
            self.fuc()
            self.__first_init = False

    # def __new__(cls, *args, **kwargs):
    #     if not cls.__instance:
    #         cls.__instance = object.__new__(cls)
    #     return cls.__instance

    @staticmethod
    def fuc():
        import ranker
        with open('ix_to_word.json', 'r') as load_f:
            words = json.load(load_f)
        with open('word_to_ix.json', 'r') as load_f:
            word_ix = json.load(load_f)
        # time_start = time.time()
        user = sim_user.SynUser()
        ranker = ranker.Ranker()
        model = NetSynUser(user.vocabSize + 1)
        if torch.cuda.is_available():
            model.cuda()
        # time_end = time.time()
        # print('t1 = ', time_end - time_start)

        batch_size = 2
        model.init_hid(batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()

        model.load_state_dict(torch.load('/home/wangshuo/work/project/fashion-retrieval/models/sl-50.pt'))
        all_input = user.test_feature
        return user, ranker, words, word_ix, model, batch_size, all_input


global user0, ranker0, words0, word_ix0, model0, batch_size0, all_input0
global user, ranker, words, word_ix, model, batch_size, all_input, idx
idx = False
# (user0, ranker0, words0, word_ix0, model0, batch_size0, all_input0) = copy.deepcopy(Model_Loader.fuc())
# (user, ranker, words, word_ix, model, batch_size, all_input) = (user0, ranker0, words0, word_ix0, model0, batch_size0, all_input0)
(user, ranker, words, word_ix, model, batch_size, all_input) = Model_Loader.fuc()
(user0, ranker0, words0, word_ix0, model0, batch_size0, all_input0) = copy.deepcopy((user, ranker, words, word_ix, model, batch_size, all_input))
model.eval()
act_img_idx = torch.LongTensor(batch_size)
ranker.update_rep(model, user.test_feature)
user.sample_idx(act_img_idx, train_mode=False)
# model.eval()

def dialog_test2(txt_input_zh, act_img_idx, init):
    global user, ranker, words, word_ix, model, batch_size, all_input, idx
    print(act_img_idx)
    new_act_input = all_input[act_img_idx]
    print(all_input)
    if torch.cuda.is_available():
        # user_input = user_input.cuda()
        # neg_input = neg_input.cuda()
        new_act_input = new_act_input.cuda()
    # user_input, neg_input, new_act_input = Variable(user_input, volatile=True), Variable(neg_input, volatile=True), Variable(new_act_input, volatile=True)
    new_act_input = Variable(new_act_input, volatile=True)
    new_act_emb = model.forward_image(new_act_input)
    # ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
    # user_emb = model.forward_image(user_input)
    # neg_emb = model.forward_image(neg_input)
    act_emb = new_act_emb

    print("我想要的跟您所提供图片不同之处在于：")

    # txt_input_zh = input("")
    # txt_input_zh = request.data
    txt_input1 = 'are ' + translate(txt_input_zh)
    txt_input0 = txt_input1.replace('-', ' ')
    # txt_input2 = ""
    txt_lower = txt_input0.lower()
    print(txt_lower)
    txt_lower0 = re.sub('[^a-zA-Z ]', '', txt_lower)
    L = txt_lower0.split()
    print(L)
    print(type(L))
    num_Tensor = np.zeros(16)
    for x in range(len(L)):
        if L[x] in word_ix.keys():
            num_Tensor[x] = word_ix[L[x]]

    txt_input0 = torch.from_numpy(np.array([num_Tensor, num_Tensor]))
    res = torch.LongTensor(batch_size, 16).zero_()
    res[0:batch_size, :].copy_(txt_input0)
    txt_input = res

    # user.sample_idx(neg_img_idx, train_mode=False)
    if torch.cuda.is_available():
        txt_input = txt_input.cuda()
    txt_input = Variable(txt_input, volatile=True)

    # print("k = ", k)
    ttt = txt_input.cpu().numpy()
    ttt = ttt.astype(np.int16)
    for batch_id in range(batch_size):
        txt_ans = ''
        for word in range(15):
            zzz = ttt[batch_id][word]
            if (zzz != 0):
                txt_ans += words[str(zzz)] + ' '
            else:
                break
        print(txt_ans)
    # print('act_emb: ', act_emb)
    print('txt_input = ', txt_input)
    # update the query action vector given feedback image and text feedback in this turn
    idx = not idx
    action = model.merge_forward(act_emb, txt_input, init, idx)
    # obtain the next turn's feedback images
    img_rank = ranker.nearest_neighbor(action.data)[1]
    print(img_rank)
    return list(img_rank)

def dialog_test(txt_input_zh, act_img_idx, rank_init):

    if rank_init == 1:
        global user, ranker, words, word_ix, model, batch_size, all_input
        (user, ranker, words, word_ix, model, batch_size, all_input) = (user0, ranker0, words0, word_ix0, model0, batch_size0, all_input0)
        # print(all_input)
        # print(all_input0)
        model.eval()
        # act_img_idx0 = [3000, 4212]
        # act_img_idx0 = torch.LongTensor(np.array(act_img_idx0))
        act_img_idx0 = torch.LongTensor(batch_size)
        # print('act_idx: ', act_img_idx0)
        ranker.update_rep(model, user.test_feature)
        user.sample_idx(act_img_idx0, train_mode=False)
        # print(act_img_idx0)
        return dialog_test2(txt_input_zh, act_img_idx0, rank_init)
    else:
        return dialog_test2(txt_input_zh, act_img_idx, rank_init)


@app.route('/register', methods=['POST'])
def register():
    print(request.data.decode('utf-8'))
    z = request.data.decode('utf-8')
    # c = json.loads(jsonify(request.data.decode('utf-8')))
    print(type(z))
    print(z)
    a = z.split(',')[0]
    b = z.split(',')[1]
    c = a.split(':')[1]
    d = int(b.split(':')[1].split('}')[0])
    e = int(z.split(',')[2].split(':')[1].split('}')[0])
    print(c, d, e)
    return str(dialog_test(c, d, e))


app.run('0.0.0.0', port=8555)
