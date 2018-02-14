import os, scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score

data_dir = './data'
features_dir = './features'

print("Loading tweet texts...")
with open(os.path.join(features_dir, 'text_processed.txt'), 'r') as f_text:
	x_text = f_text.readlines()

print("Loading tweet user descriptions...")
with open(os.path.join(features_dir, 'description_processed.txt'), 'r') as f_description:
	x_description = f_description.readlines()

print("Loading tweet hashtags...")
with open(os.path.join(features_dir, 'hashtags_processed.txt'), 'r') as f_hashtags:
	x_hashtags = f_hashtags.readlines()

with open(os.path.join(data_dir, 'labels.txt'), 'r') as f_labels:
	y = np.array(f_labels.readlines())

print("Extract features...")
x_text_feats = TfidfVectorizer().fit_transform(x_text)
x_description_feats = TfidfVectorizer().fit_transform(x_description)
x_hashtags_feats = TfidfVectorizer().fit_transform(x_hashtags)
print("Dimension of text features:")
print(x_text_feats.shape)
print("Dimension of description features:")
print(x_description_feats.shape)
print("Dimension of hashtag features:")
print(x_hashtags_feats.shape)

print("Start training and predict...")
kf = KFold(n_splits=10)
fout = open(os.path.join(data_dir, 'integrated_fitted_classes.txt'), 'w')
text_pred = np.empty([0, 10])
description_pred = np.empty([0, 10])
hashtag_pred = np.empty([0, 10])

for train, test in kf.split(x_text_feats):
	model = MultinomialNB().fit(x_text_feats[train], y[train])
	predicts = model.predict_proba(x_text_feats[test])
	text_pred = np.concatenate((text_pred, predicts))

print(text_pred)
print(text_pred.shape)


for train, test in kf.split(x_description_feats):
	model = MultinomialNB().fit(x_description_feats[train], y[train])
	predicts = model.predict_proba(x_description_feats[test])
	print(predicts.shape)
	description_pred = np.concatenate((description_pred, predicts))

print(description_pred)
print(description_pred.shape)


for train, test in kf.split(x_hashtags_feats):
	model = MultinomialNB().fit(x_hashtags_feats[train], y[train])
	predicts = model.predict_proba(x_hashtags_feats[test])
	hashtag_pred = np.concatenate((hashtag_pred, predicts))

print(hashtag_pred)
print(hashtag_pred.shape)

## Generate multiple sets of weights for text, description, and hashtags
w_text_range = np.arange(0.3, 0.6, 0.05)
w_description_range = np.arange(0, 0.3, 0.1)
weight_sets = []

for w_description in w_description_range:
	for w_text in w_text_range:
		weight_sets.append([w_text, w_description, 1 - w_text - w_description])

