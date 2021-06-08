#*- coding:utf-8 -*- coding

import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from numba import jit
from gensim.models import KeyedVectors
import pandas as pd
import joblib
class word2prob(object):

    def __init__(self,word_count_file_path):
        file_path = word_count_file_path
        with open(file_path, 'r',encoding='utf-8') as f:
            self.prob = json.load(f)
            total = sum(self.prob.values())


        self.prob = {k: (self.prob[k] / total) for k in self.prob}
        self.min_prob = min(self.prob.values())
        self.count = total

    def __getitem__(self, w):
        return self.prob.get(w.lower(), self.min_prob)

    def __contains__(self, w):
        return w.lower() in self.prob

    def __len__(self):
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())

class word2vec(object):

    def __init__(self,embedding_path):

        self.embedding_path = embedding_path
        vectors = KeyedVectors.load(embedding_path, mmap='r+')    #Vectors(name=embedding_path)
        self.vectors = vectors

    def __getitem__(self, w):
        return self.vectors[w]

    def __contains__(self, w):
        return w in self.vectors

class uSIF(object):
    def __init__(self, vec, prob, n=11, m=5):

        self.vec = vec
        self.m = m

        if not (isinstance(n, int) and n > 0):
            raise TypeError("n should be a positive integer")

        vocab_size = float(len(prob))
        threshold = 1 - (1 - 1 / vocab_size) ** n
        alpha = len([w for w in prob.vocab() if prob[w] > threshold]) / vocab_size
        Z = 0.5 * vocab_size
        self.a = (1 - alpha) / (alpha * Z)

        self.weight = lambda word: (self.a / (0.5 * self.a + prob[word]))

    @jit
    def _to_vec(self, sentence):
        tokens = sentence.split()
        #v_t = np.zeros(300)
        if tokens == []:
            return np.zeros(300) + self.a
        else:
            # for i in tokens:
            #     v_t +=self.vec[i]*self.weight(i)/len(tokens)
            # return v_t
            v_t = np.array([self.vec[t] for t in tokens])
            v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
            v_t = np.array([self.weight(t) * v_t[i, :] for i, t in enumerate(tokens)])
            return np.mean(v_t, axis=0)

    def embed(self, sentences):
        sentences_vectors = self._to_vec(sentences).reshape(1,-1)
        #sentences_vectors = [self._to_vec(s) for s in sentences.split()]
        if self.m == 0:
            return sentences_vectors
        proj = lambda a, b: a.dot(b.transpose())*b
        svd = TruncatedSVD(n_components=self.m, n_iter=7,random_state=0).fit(sentences_vectors)
        #save model
        joblib.dump(svd, 'svd.pkl')