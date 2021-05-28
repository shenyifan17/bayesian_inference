import numpy as np
import matplotlib.pyplot as plt
import math
import random
from random import randrange
import collections

def cleanTextFullSymbols(text):
    replace_list = ['%','@']
    for ch in replace_list:
        text = text.replace(ch,'')
    text = text.lower()
    space_list = ['  ','   ','    ','     ','      ','       ','        ']
    for ch in space_list:
        text = text.replace(ch,' ')
    text = text.replace('  ',' ')
    text = text.replace('\n', ' ')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('‘', "'")
    text = text.replace('’', "'")
    text = text.replace("\'", "")
    text = text.replace('á', 'a')
    text = text.replace('é', 'e')
    text = text.replace('ë', 'e')
    text = text.replace('è', 'e')
    text = text.replace('í', 'i')
    text = text.replace('ó', 'o')
    return text

def randomCoin(p):
    uniform_prob = random.uniform(0,1)
    if uniform_prob < p:
        return True
    else:
        return False

def propSwap(key):
#   key: list
    key_new = [ch for ch in key]
    n = len(key)
    i = randrange(0,n-1)
    j = randrange(0,n-1)
    if i == j:
        propSwap(key)
    jj = key_new[j]
    ii = key_new[i]
    key_new[i] = jj
    key_new[j] = ii
    return key_new
#   key_new: list


def dcyptMsg(key,symbols,message):
#     key: list
#     symbols: list
#     message: string
    dcypted_msg = ''
    m = len(message)
    for i in range(m):
        dcypted_msg += key[symbols.find(message[i])]
    return dcypted_msg
#     new_message: string

def generateEquMtx(war_and_peace,symbols):
# war_and_peace: string (full text, very long)
# symbols: string
    n = len(symbols)
    count_mtx = np.zeros((n,n))
    k = len(war_and_peace)
#   count
    for i in range(k):
        where_1 = symbols.find(war_and_peace[i-1])
        where_2 = symbols.find(war_and_peace[i])
#       check whether they are in the symbols (in case of special letter)
        if (where_1 != -1) and (where_2 != -1):
            count_mtx[where_1,where_2] += 1
        else:
            count_mtx[where_1,where_2] += 0
#   normalise
    normal = np.sum(count_mtx,axis = 1)
    normal_re = normal.reshape(-1,1)
    normal_rept = np.repeat(normal_re,n,axis = 1)
    equ_mtx = np.nan_to_num(count_mtx / normal_rept)

    return equ_mtx

def evaluateDetailedBalance(key,symbols,message,equ_mtx):
    dcypted_msg = dcyptMsg(key,symbols,message)
    n = len(message)
    eva = 0
    for i in range(n):
        eva += math.log(equ_mtx[symbols.find(dcypted_msg[i-1]),symbols.find(dcypted_msg[i])])
    return eva

def metropolisHastingDecrypt(n_iter,message,equ_mtx,symbols):
    key = list(symbols)
    decrypt_key = []
    for i in range(n_iter):
        this_key = key
        swap_key = propSwap(key)
        this_eva = evaluateDetailedBalance(this_key,symbols,message,equ_mtx)
        swap_eva = evaluateDetailedBalance(swap_key,symbols,message,equ_mtx)
        accept_prob = min(math.exp(swap_eva - this_eva),1)
        if (randomCoin(accept_prob)) and (swap_eva > this_eva):
            key = swap_key
            decrypt_key = swap_key
        if i%5000 == 0:
            print('============================================================================================')
            print('iteration',i,': \n ',dcyptMsg(decrypt_key,symbols,message)[0:600])
            print('----------------------------------------')
            print('log likelihood for current mapping: ', this_eva)
            print('log likelihood for proposed swap mapping:', swap_eva)
#             print ('iter',i,':',decrypt_key)
    return decrypt_key

