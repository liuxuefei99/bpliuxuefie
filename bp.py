# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:33:35 2019

@author: Liter Frye
"""
import numpy as np
import matplotlib.pyplot
from numpy import random as nr 

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

#训练神经网络
def learning(all_value,w_input2hide,w_hide2output):
    all_value=all_value
    w_input2hide=w_input2hide
    w_hide2output=w_hide2output
#确定输入与输出    
    input_pic=np.array(all_value[1:],dtype='float64').reshape(28,28)
    input_pic=input_pic/255*0.99+0.01
    output_pic=np.full([10,1],0.1,dtype='float64')
    output_pic[int(all_value[0]),0]=0.99
    
#计算隐藏层与输出层    
    x_hide=np.dot(w_input2hide,input_pic.reshape(28*28,1))
    o_hide=sigmoid(x_hide)
    x_output=np.dot(w_hide2output,o_hide)
    o_output=sigmoid(x_output)
#计算反向误差：主要为输出层误差与隐藏层误差    
    
    error_output=output_pic-o_output
    error_hide=np.dot(np.transpose(w_hide2output),error_output)
    #error_input=np.dot(np.transpose(w_input2hide),error_hide)
#更新权重    
    
    w_hide2output=w_hide2output+a*np.dot((error_output*o_output*(1-o_output)),np.transpose(o_hide))
    w_input2hide=w_input2hide+a*np.dot((error_hide*o_hide*(1-o_hide)),np.transpose(input_pic.reshape(28*28,1)))
#返回更新的权重
    return w_hide2output,w_input2hide

def testing(data_list_test):
    data_list_test=data_list_test
    true=0
    wrong=0
    for i in range(0,len(data_list_test)):
        all_value=data_list_test[i].split(',')
        input_pic=np.array(all_value[1:],dtype='float64').reshape(28,28)
        input_pic=input_pic/255*0.99+0.01
        x_hide=np.dot(w_input2hide,input_pic.reshape(28*28,1))
        o_hide=sigmoid(x_hide)
        x_output=np.dot(w_hide2output,o_hide)
        o_output=sigmoid(x_output)
        if int(all_value[0])==np.argmax(o_output):
            true=true+1
        else:
            wrong=wrong+1
    accuracy=true/(true+wrong)
    return accuracy
    

data_file=open("C:\\Users\\Liter Frye\\Desktop\\mnist_train.csv",'r')
data_list_train=data_file.readlines()
data_file.close()
data_file=open("C:\\Users\\Liter Frye\\Desktop\\mnist_test.csv",'r')
data_list_test=data_file.readlines()
data_file.close()

#先确定权重集，有100个隐藏节点,学习率为0.1，随机初始化权重
hide_pot=100
a=0.1
w_input2hide=nr.rand(hide_pot,784)-0.5
w_hide2output=nr.rand(10,hide_pot)-0.5

#训练2世代
for j in (0,2):
    for i in range(0,len(data_list_train)):    
        all_value=data_list_train[i].split(',')
        w_hide2output,w_input2hide=learning(all_value,w_input2hide,w_hide2output)
#计算准确率    
accuracy=testing(data_list_test)




