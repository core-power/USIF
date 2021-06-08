import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库
import re
import json
def read_file(fpath):
    Block_Size = 1024
    with open(fpath,"r",encoding='utf-8') as f:
        while True:
            block = f.read(Block_Size)
            if block:
                yield block
            else:
                return

print('载入数据')
# with open('word.txt',"r",encoding='utf-8') as f:
#     data = f.readlines()        # 读取文件

def func_dict():
    """
    方法一：使用字典
    :param word_list:
    :return:
    """
    count_dict = {}

    #for item in word_list:
    with open('word.txt', "r", encoding='utf-8') as f:
        item = f.readline()
        while item:
            item = re.sub('\n','',item)
            count_dict[item] = count_dict[item] + 1 if item in count_dict else 1
            item = f.readline()  # 读取文件
    return count_dict
with open('word_count.json','w',encoding='utf-8') as f:
    word_counts = func_dict()
    f.write(json.dumps(dict(word_counts),ensure_ascii=False,indent=0))
print('数据载入完成')




