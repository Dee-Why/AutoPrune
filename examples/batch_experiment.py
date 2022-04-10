#encoding:utf-8
import pickle
import contextlib
import os
import prune_conv_mnist

# TODO: 实现多次重复实验，并保存log到多个文件，并且提取每一次重复实验的结果数据录入表格中

dirs = 'experiment/conv_fc/batch0/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

result_list = []
for i in range(5):
    file_path = dirs + 'conv_fc_fashion' + str(i)
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            result = prune_conv_mnist.exp()
            result_list.append(result)

 
with open(dirs+'result_list.pkl','wb') as p:
    pickle.dump(result_list,p)   #将列表t保存起来
 
 
# with open(dirs+'result_list.pkl','rb') as r:
#     a = pickle.load(r)  #将列表读取