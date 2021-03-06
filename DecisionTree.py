﻿# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:00:36 2017
@author: xiao
"""

# ID3决策树的实现
# 使用信息增益进行计算
# D 是一个数据框，A 是一个字典（其中包括属性和属性的取值）
# 决策树重要的是一棵树，即结点的选择
from pandas import Series,DataFrame
import numpy as np
# 检查D在属性A上的取值是否一致，一致则返回True,否则返回False
def check_unique(D,A):
    for i in A:
        if len(set(D[i])) != 1:
            return False
    return True
# 计算信息熵
def Ent(D,label):
    ent = 0
    for i in set(D[label]):
        ent += -(sum(D[label] == i)/len(D) * np.log2(sum(D[label] == i)/len(D)))
    return ent
# 返回D在属性a上的一系列信息熵
def Ent_n(D,A,label,a):
    var = []
    for i in A[a]:
        var.append(Ent(D[D[a] == i],label))
    return np.array(var)
# 返回D在属性a上的子集占比
def P(D,A,a):
    p = []
    for i in A[a]:
        p.append(sum(D[a] == i)/len(D))
    return np.array(p)
# 返回计算C4.5需要的系数
def IV(D,A,a):
    iv = 0
    for i in A[a]:
        iv += -(sum(D[a] == i)/len(D)*np.log2(sum(D[a] == i)/len(D)))
    return iv
# 选择最佳属性--ID3算法  
def best_gain_var(D,A,label):
    var = []
    gain = []
    for a in A.keys():
        gain.append(Ent(D,label) - sum(P(D,A,a)*Ent_n(D,A,label,a)))
        var.append(a)
    return var[gain.index(max(gain))]
# 选择最佳属性--C4.5算法
def best_gainratio_var(D,A,label):
    var = []
    gain = []
    for a in A.keys():
        gain.append((Ent(D,label) - sum(P(D,A,a)*Ent_n(D,A,label,a)))/IV(D,A,a))
        var.append(a)
    return var[gain.index(max(gain))]
# 选择父节点类别占多的类        
def mylabel(D,label):
    mark = []
    num = []
    for i in set(D[label]):
        num.append(list(D[label]).count(i))
        mark.append(i)
    my_label = mark[num.index(max(num))]
    return my_label
               
def ID3_tree(D,A,label,node):
    if set(D[label]) == 1:   
        node[label] = set(D[label])
        return node
    if (not any(A)) or check_unique(D,A):
        node[label] = mylabel(D,label)#max(list(D.label).count(x) for x in set(D.label))
        return node
    a_start = best_gain_var(D,A,label) # 选出纯度最高的属性，即信息增益最高的属性
    node1 ={}
    
    for i in set(A[str(a_start)]):
        if sum(D[a_start] == i) == 0:
            node[label] = mylabel(D,label)#max(list(D.label).count(x) for x in set(D.label))
            return node
        else:
            D1 = D[D[a_start] == i]
            A1 = A.copy()
            A1.pop(a_start)
            node1[i] = {}
            node1[i] = ID3_tree(D1,A1,label,node1[i])
    node[a_start] = node1
    return node   
    
def C45_tree(D,A,label,node):
    if set(D[label]) == 1:   
        node[label] = set(D[label])
        return node
    if (not any(A)) or check_unique(D,A):
        node[label] =  mylabel(D,label)# max(list(D.label).count(x) for x in set(D.label))
        return node
    a_start = best_gainratio_var(D,A,label) # 选出纯度最高的属性，即信息增益最高的属性
    node1 ={}
    
    for i in set(A[str(a_start)]):
        if sum(D[a_start] == i) == 0:
            node[label] = mylabel(D,label)#max(list(D.label).count(x) for x in set(D.label))
            return node
        else:
            D1 = D[D[a_start] == i]
            A1 = A.copy()
            A1.pop(a_start)
            node1[i] = {}
            node1[i] = C45_tree(D1,A1,label,node1[i])
    node[a_start] = node1
    return node     
    
# 数据节点的预测就是不断地去提取子树
def decision_tree_pre(node,D,label):
    L = []
    for d in range(0,len(D)):
        val = D.ix[d]
        node1 = node.copy()
        while 'node' not in node1:
            for (i,v) in node1.items():
                node1 = v[D.ix[d][i]]
        L.append(node1[label])
    return L
        
    
    
if __name__ == '__main__':
    D_train = {'色泽':['青绿','乌黑','乌黑','青绿','乌黑','青绿','浅白','乌黑','浅白','青绿'],
               '根蒂':['蜷缩','蜷缩','蜷缩','稍蜷','稍蜷','硬挺','稍蜷','稍蜷','蜷缩','蜷缩'],
               '敲声':['浊响','沉闷','浊响','浊响','浊响','清脆','沉闷','浊响','浊响','沉闷'],
               '纹理':['清晰','清晰','清晰','清晰','稍糊','清晰','稍糊','清晰','模糊','稍糊'],
               '脐部':['凹陷','凹陷','凹陷','稍凹','稍凹','平坦','凹陷','稍凹','平坦','稍凹'],
               '触感':['硬滑','硬滑','硬滑','软粘','软粘','软粘','硬滑','软粘','硬滑','硬滑'],
               '标签':['是','是','是','是','是','否','否','否','否','否']}
    D_test = { '色泽':['青绿','浅白','乌黑','乌黑','浅白','浅白','青绿'],
               '根蒂':['蜷缩','蜷缩','稍蜷','稍蜷','硬挺','蜷缩','稍蜷'],
               '敲声':['沉闷','浊响','浊响','沉闷','清脆','浊响','浊响'],
               '纹理':['清晰','清晰','清晰','稍糊','模糊','模糊','稍糊'],
               '脐部':['凹陷','凹陷','稍凹','稍凹','平坦','平坦','凹陷'],
               '触感':['硬滑','硬滑','硬滑','硬滑','硬滑','软粘','硬滑'],
               '标签':['是','是','是','否','否','否','否']}
    label = '标签'
    A = {}
    for i in D_train.keys():
        if i != label:
            A[i] = set(D_train[i])
    D_train = DataFrame(D_train)
    D_test = DataFrame(D_test)
    node = {}
    node = ID3_tree(D_train,A,label,node)
    node1 = {}
    node1= C45_tree(D_train,A,label,node1)

# 这里数据训练过小，数据学习过差，如果想要提高算法模型，可以换一下数据集
# PS:决策树算法模型还没有对其进行剪枝

