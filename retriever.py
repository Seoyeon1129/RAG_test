from konlpy.tag import Okt

class SparseRetriever():
    def __init__(self, corpus, tokenizer=Okt()):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.tokenized_corpus = [tokenizer.morphs(text) for text in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    def retrieve(self, query, k=5):
        tokenized_query = self.tokenizer.morphs(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return results