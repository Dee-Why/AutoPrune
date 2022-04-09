#encoding:utf-8
import pickle
import contextlib
import os
import prune_fc_mnist

# TODO: 实现多次重复实验，并保存log到多个文件，并且提取每一次重复实验的结果数据录入表格中

dirs = 'experiment/batch3/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

result_list = []
for i in range(5):
    file_path = dirs + 'fcn2_fashion_lth' + str(i)
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            result = prune_fc_mnist.exp()
            result_list.append(result)

 
with open(dirs+'result_list.pkl','wb') as p:
    pickle.dump(result_list,p)   #将列表t保存起来
 
 
# with open(dirs+'result_list.pkl','rb') as r:
#     a = pickle.load(r)  #将列表读取
 
dirs = 'experiment/batch4/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

result_list = []
for i in range(5):
    file_path = dirs + 'fcn2_fashion_lth' + str(i)
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            result = prune_fc_mnist.exp()
            result_list.append(result)

 
with open(dirs+'result_list.pkl','wb') as p:
    pickle.dump(result_list,p)   #将列表t保存起来