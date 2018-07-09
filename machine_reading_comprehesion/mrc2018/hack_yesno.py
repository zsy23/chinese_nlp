# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def gen_yesno(sample, a):
    yesno = {
        'Depends': 0,
        'Yes': 1,
        'No': 2
    }

    neg = ['无', '不', '没', '别', '假', '骗']

    q_seg = sample['question_tokens']
    q_seg = [w.decode('utf-8') for w in q_seg]
    a = a.decode('utf-8')
    q = ''.join(q_seg)
    
    guess = 0
    if guess != 2 and \
        '真' in q and ('假' in a or '伪' in a) or \
        '假' in q and ('真' in a or '正品' in a):
        guess = 2
    if guess != 2 and \
        ('谣言' in a or '纯属' in a or '难' in a or '垃圾' in a or \
        '坑' in a or '一般般' in a or '否定' in a or '伪劣' in a or \
        '否定' in a):
        guess = 2
    if guess != 2 and len(a) < 32:
        if '骗' not in q and '骗' in a:
            guess = 2
        for n in ['无', '不', '没', '别', '假']:
            if a.count(n) == 1:
                guess = 2
                break
    if guess != 2 and '不' in q:
        pos = q.find('不')
        if pos-1>=0 and pos+1<len(q) and \
            q[pos-1] == q[pos+1] and q[pos-1] in a:
            pos2 = a.find(q[pos-1])
            if pos2-1>=0 and a[pos2-1] == '不':
                guess = 2
    if guess != 2:
        for w in q_seg:
            if w in a:
                qpos = q.find(w)
                apos = a.find(w)                     
                qneg = sum([1 for c in q[max(0, qpos-9):qpos+len(w)+1] if c in neg]) % 2
                if '不' in q[max(0, qpos-9):qpos+len(w)+1]:
                    pos = max(0, qpos-9) + q[max(0, qpos-9):qpos+len(w)+1].find('不')
                    if pos-1>=0 and pos+1<len(q) and q[pos-1] == q[pos+1]:
                        qneg = 1 - qneg
                aneg = sum([1 for c in a[max(0, apos-9):apos+len(w)+1] if c in neg]) % 2

                if qneg ^ aneg:
                    guess = 2
                    break
    if '不一定' in a or '不完全' in a:
        guess = 0
    if '不错' in a or '没问题' in a or '没事' in a:
        guess = 1
    if '骗' in q and '骗' in a and '不' not in a:
        guess = 1
    if '不' not in a[:5]:
        for c in ['可以']:
            if c in a[:5]:
                guess = 1
                break
    if a.find('能') == 0 or a.find('是的') == 0 or \
        a.find('有的') == 0 or a.find('可以') == 0 or \
        a.find('会') == 0 or a.find('真的') == 0:
        guess = 1
    if len(a) >= 2 and a[1] in [',', '，', '.', '。'] or \
        len(a) >= 3 and a[0] not in neg and a[2] in [',', '，', '.', '。'] or \
        len(a) == 1 and a[0] not in neg or \
        len(a) == 2 and a[0] not in neg:
        guess = 1
    if guess == 0 and \
        '如果' not in a and '不一定' not in a and '不完全' not in a and \
        '怎么' not in a and '要看' not in a and '看是' not in a and \
        '看情况' not in a and '人而异' not in a and '慎用' not in a and \
        '推敲' not in a and '试用' not in a and '是否' not in a and \
        '不确定' not in a and '除非' not in a and '看你' not in a and \
        '看个人' not in a:
        guess = 1

    res = ''
    if guess == 0:
        res = 'Depends'
    elif guess == 1:
        res = 'Yes'
    elif guess == 2:
        res = 'No'

    return res
