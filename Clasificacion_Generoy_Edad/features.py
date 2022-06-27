from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Capitals(BaseEstimator, TransformerMixin):
        # feature that counts capitalized characters in a tweet
        def fit(self, X, Y=None):
                return self
        def transform(self, X):
                return [[sum(1 for ch in doc if ch.isupper())] for doc in X]

class Patterns(BaseEstimator, TransformerMixin):
        # feature that counts occurences for a range of patterns
        def __init__(self, patterns):
                self.patterns = patterns
        def fit(self, X, Y=None):
                return self
        def transform(self, X):
                return [[doc.lower().count(pattern)/len(doc) for pattern in self.patterns] for doc in X]

class lsa(BaseEstimator, TransformerMixin):
        def __init__(self, trunc):
                self.val=0
                self.tfidf=TfidfVectorizer()
                self.lsa=TruncatedSVD(trunc)
                self.topics_no=trunc
        def fit(self, X, Y=None):
                return self
        def transform(self, X):

                self.val=self.val+1
                tfidf=self.tfidf
                lsa=self.lsa
                if self.val == 1:
                        tfidf_vector=tfidf.fit_transform(X)
                        lsa_vec=lsa.fit_transform(tfidf_vector)
                else:
                        tfidf_vector=tfidf.transform(X)
                        lsa_vec=lsa.transform(tfidf_vector)
                return lsa_vec

class chaviza(BaseEstimator, TransformerMixin):
        # feature that counts occurences for a range of patterns
        def __init__(self, chaviza):
                self.chaviza = chaviza
        def fit(self, X, Y=None):
                return self
        def transform(self, X):
                return [[doc.lower().count(chaviza)/len(doc) for chaviza in self.chaviza] for doc in X]

