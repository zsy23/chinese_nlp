# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from collections import Counter

filter_words = {'question': ['什么', '怎', '何', '哪', '是不是', '是否', '对不对',
                             '多长', '多久', '多远', '多少', '多高', '多大', '几',
                             '有没有', '吗', '呢', '谁', '嘛'],
                'stop': ['的', '了', '和', '是', '就', '都', '而', '及', '或', '中',
                         '之', '才', '与', '多'],
                'punctuation': [',', '，', '.', '。', '?', '？', '!', '！',
                                ':', '：', ';', '；']}

# question-answer pair
ques_keys = {
    'time': {
        'keys': ['时候', '时间'],
        'ans': ['月', '时间', '年', '星期', '号']
    },
    'place': {
        'keys': ['哪', '什么地方', '地点'],
        'ans': ['地点']
    },
    'people': {
        'keys': ['谁', '叫什么', '名字'],
        'ans': ['叫', '是']
    },
    'money': {
        'keys': ['价格', '价钱', '钱', '工资', '费用', '花费', 
                '金额', '值'],
        'ans': ['块', '元', '十', '千', '百', '万', '$']
    },
    'height': {
        'keys': ['多高', '多长', '多宽', '身高', '距离', '多远'
                '高多少', '长度', '长多少', '宽度', '宽多少'],
        'ans': ['m', '米', '里']
    },
    'weight': {
        'keys': ['多重', '体重'],
        'ans': ['g', '斤', '克']
    },
    'amount': {
        'keys': ['多少', '几', '数目', '数量'],
        'ans': ['个']
    }
}


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False

def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False

def ques_ans_pair(q, p):
    score = 2.0

    for k in ques_keys['time']['keys']:
        if k in q:
            for a in ques_keys['time']['ans']:
                if a in p:
                    return score

    for k in ques_keys['place']['keys']:
        if k in q:
            for a in ques_keys['place']['ans']:
                if a in p:
                    return score  

    for k in ques_keys['people']['keys']:
        if k in q:
            for a in ques_keys['people']['ans']:
                if a in p:
                    return score  

    for k in ques_keys['money']['keys']:
        if k in q:
            for a in ques_keys['money']['ans']:
                if a in p:
                    return score  

    for k in ques_keys['height']['keys']:
        if k in q:
            for a in ques_keys['height']['ans']:
                if a in p:
                    return score  

    for k in ques_keys['weight']['keys']:
        if k in q:
            for a in ques_keys['weight']['ans']:
                if a in p:
                    return score  

    for k in ques_keys['amount']['keys']:
        if k in q:
            for a in ques_keys['amount']['ans']:
                if a in p:
                    return score  

    return 1.0

def select_paragraph(q, paras):
    q = [w.decode('utf-8') for w in q]
    for idx in range(len(paras)):
        p = paras[idx]
        paras[idx] = [w.decode('utf-8') for w in p]

    # filter question words, stop words and other words
    keywords = []
    for w in q:
        flag = True
        for c in filter_words['question']:               
            if c in w:
                flag = False
                break
        if flag:
            keywords.append(w)
    keywords = [w for w in keywords if w not in filter_words['stop']]
    keywords = [w for w in keywords if len(w) > 1 or (len(w) == 1 and not is_other(w.decode('utf-8')))]

    para_infos = []
    for p in paras:
        if len(p) == 0:
            continue

        para_cnt = Counter([w.lower() for w in p])

        # remove some first para
        if len(para_infos) == 0:
            if '小编' in ''.join(p):
                para_infos.append((len(para_infos), 0.0))
                continue
            other_cnt = 0
            idx = -1
            flag = False
            while -idx <= len(p) and other_cnt < 2:
                if p[idx] == '?' or p[idx] == '？' or \
                    p[idx] == '介绍' or p[idx] == '下面':
                    flag = True
                    break
                if flag:
                    break
                if is_other(p[idx].decode('utf-8')):
                    other_cnt += 1    
                idx -= 1
            if flag:
                para_infos.append((len(para_infos), 0.0))
                continue

        # like '1...2...3..' or '一...二...三...' must be most related
        first = 0
        content = ''
        for idx in range(len(p)):
            w = p[idx]
            if first == 0:
                if '1' == w:
                    first = 1
                elif '一' == w:
                    first = 2
                elif '首先' == w:
                    first = 3
                elif 'a' == w:
                    first = 5
            elif first == 1 and '2' == w:
                first = 4
                break
            elif first == 2 and '二' == w:
                first = 4
                break
            elif first == 3 and ('然后' == w or '接着' == w or '最后' == w):
                first = 4
                break
            elif first == 5 and 'b' == w:
                first = 4
                break
            elif first > 0:
                content += ''.join(w)
        score = 1.0   
        if '病情分析' in ''.join(p) or \
            '问题分析' in ''.join(p):
            score = 3.0          
        if first == 4 and len(content) > 2:
            score = 10.0

        score *= ques_ans_pair(''.join(q), ''.join(p))

        frequency = sum([para_cnt[w.lower()] for w in keywords])
        if float(frequency) / len(p) > 0.5:
            para_infos.append((len(para_infos), 0.0))
            continue

        para_len = sum([len(w) for w in p])
        para_infos.append((len(para_infos), score * (frequency + 0.5) * para_len))
    
    para_infos.sort(key=lambda x: (-x[1], x[0]))
    guess = -1
    if len(para_infos) > 0:
        guess = para_infos[0][0]  

    return guess