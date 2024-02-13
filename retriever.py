from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
import streamlit as st

class SparseRetriever():
    def __init__(self, corpus, tokenizer=Okt()):
        self.tokenizer = tokenizer
        self.corpus = corpus
        # self.tokenized_corpus = [tokenizer.morphs(text) for text in corpus]
        self.tokenized_corpus = list()
        n = len(corpus)
        my_bar = st.progress(0. , text=f'0 / {n}')
        for i, text in enumerate(corpus):
            my_bar.progress((i+1)/n, text=f'{i+1} / {n}')
            self.tokenized_corpus.append(tokenizer.morphs(text))
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        my_bar.empty()
    def retrieve(self, query, k=5):
        tokenized_query = self.tokenizer.morphs(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return results