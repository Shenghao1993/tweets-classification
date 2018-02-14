# encoding=utf8

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score


data_dir = './data' 

print("Loading data...")
with open(os.path.join(data_dir, 'samples_processed.txt'), 'r') as f:
	x = f.readlines()
with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
	y = np.array(f.readlines())


print("Extract features...")
x_feats = TfidfVectorizer().fit_transform(x)
print(x_feats.shape)


print("Start training and predict...")
kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
for train, test in kf.split(x_feats):
	model = MultinomialNB().fit(x_feats[train], y[train]) 
	predicts = model.predict(x_feats[test])
	print(classification_report(y[test],predicts))
	avg_p	+= precision_score(y[test],predicts, average='macro')
	avg_r	+= recall_score(y[test],predicts, average='macro')

print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))