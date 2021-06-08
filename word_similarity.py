import warnings
warnings.filterwarnings("ignore")
import jieba
import numpy as np
jieba.load_userdict("jieba_word.txt")
from usif import word2prob,word2vec,uSIF
from sklearn.metrics import average_precision_score
from gensim.models import KeyedVectors
with open('test2.tsv','r',encoding='utf-8') as f:
    data = f.readlines()
f.close()
sentences_1 = []
sentences_2 = []
target = []
for i in data:
    text = i.split('\t')
    sentences_1.append(' '.join(jieba.lcut(text[0])))
    sentences_2.append(' '.join(jieba.lcut(text[1])))
    target.append(float(text[2].replace('\n','')))
vec = word2vec('E:/微信文本语料/bigram/wechat_bigram.char')
prob = word2prob('word_count.json')
my_usif = uSIF(vec,prob,m=5)
def word_average(sentence):
    sent = sentence.split()
    vector = np.zeros(300)
    for i in range(len(sent)):
        try:
            vector +=vec[sent[i]]/len(sent)
        except:continue
    return vector
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
sim = []
for j in range(len(target)):
    cos = cos_sim(my_usif.embed(sentences_1[j]),my_usif.embed(sentences_2[j]))
    sim.append(cos)
print("准确率为：",average_precision_score(target, sim))

