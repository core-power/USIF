# USIF
usif 算法通过使得语义向量空间平滑，并且消除句向量点之间的方差。


相似度计算在：word_similarity.py


在使用usif的时候需要先在大语料中训练word2vec或者fasttext词向量模型，然后在统计词频，且需要算出大语料中共现矩阵前五的奇异矩阵模型。


vec = word2vec('../bigram/wechat_bigram.char')


prob = word2prob('word_count.json')


my_usif = uSIF(vec,prob,m=5)



知乎文章链接：https://zhuanlan.zhihu.com/p/263511812