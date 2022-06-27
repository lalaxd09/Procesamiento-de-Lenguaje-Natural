import os
import sys
import nltk
from lxml import etree
from features import *
from sklearn import svm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD

def main(argv):
	trainingdir = argv[1]
	testdir = argv[2]
	languages = ['dutch', 'english', 'italian', 'spanish']
        
	for language in languages:
		if language == 'spanish':
			# train the system
			Xtrain, Ytraingender, Ytrainage = [], [], []
			traintweets = loadData(trainingdir, language)
			traintruths = loadTruth(trainingdir, language)
			for author, tweet in traintweets.items():
				Xtrain.extend(tweet)
				##015f2a45-47f5-48bf-904c-264acc3475df:::F:::18-24:::0.1:::0.1:::0.4:::0.2:::0.1
				#truths[split[0]]=(split[1],split[2])
				Ytraingender.extend([traintruths[author][0]] * len(tweet))
				Ytrainage.extend([traintruths[author][1]] * len(tweet))
			
			X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrainage, test_size=0.2, random_state=0)
			
			# features = FeatureUnion([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,4))),
									 #('patterns', Patterns(['.','!','?','rt','#','@', 'http']))
									 
								#])
			
			features = FeatureUnion([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,4))),
									 ('patterns', Patterns(['.','!','?','rt','#','@', 'http'])),
									 ],
									 transformer_weights={'tfidf': 3})
			
			#Train model
#Train model

			clf = LogisticRegression(max_iter = 10000)
			pipeline = features.fit(X_train,y_train).transform(X_train)
			
			
			#Test model
			vectorizer = TfidfVectorizer()
			vectors_train = vectorizer.fit_transform(X_train)
			
			svd = TruncatedSVD(500)
			pipeline = Pipeline([("features", pipeline), ("svm", svd)])
			vectors_test = vectorizer.transform(X_test)


			
			pipeline.fit(vectors_train, y_train)
			pipeline.predict(vectors_test)



def loadData(directory, language):
	# returns a dictionary of tweets per author
	tweets = {}
	languagedir = os.path.join(directory)
	for filename in os.listdir(languagedir):
		if filename.endswith('xml'):
			handle = open(os.path.join(languagedir, filename), 'rb')
			tree = etree.fromstring(handle.read())
			documents = tree.xpath('//document')
			tweets[filename[:-4]] = [doc.text.rstrip() for doc in documents]
		
	return tweets

def loadTruth(directory, language):
	#015f2a45-47f5-48bf-904c-264acc3475df:::F:::18-24:::0.1:::0.1:::0.4:::0.2:::0.1
	# returns a dictionary of truth values per author
	truths = {}
	filepath = os.path.join(directory,  'truth.txt')
	handle = open(filepath, 'r')
	for line in handle:
		split = line.split(':::')
		truths[split[0]]=(split[1],split[2])
	
	return truths
        

if __name__ == '__main__':
	main(sys.argv)
