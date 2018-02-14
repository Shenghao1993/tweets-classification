import os, scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score

data_dir = './data'

print("Loading tweet texts...")
with open(os.path.join(data_dir, 'text_processed.txt'), 'r') as f_text:
	x_text = f_text.readlines()

print("Loading tweet user descriptions...")
with open(os.path.join(data_dir, 'description_processed.txt'), 'r') as f_description:
	x_description = f_description.readlines()

print("Loading tweet hashtags...")
with open(os.path.join(data_dir, 'hashtags_processed.txt'), 'r') as f_hashtags:
	x_hashtags = f_hashtags.readlines()
	void_ids = [i for i, tag in enumerate(x_hashtags) if tag == '\n']
	x_hashtags_new = [tag for j, tag in enumerate(x_hashtags) if j not in void_ids]

with open(os.path.join(data_dir, 'labels.txt'), 'r') as f_labels:
	y = np.array(f_labels.readlines())
	y_new = np.array([label for k, label in enumerate(y) if k not in void_ids])

print("Extract features...")
x_text_feats = TfidfVectorizer().fit_transform(x_text)
x_description_feats = TfidfVectorizer().fit_transform(x_description)
x_hashtags_feats = TfidfVectorizer().fit_transform(x_hashtags_new)
print("Dimension of text features:")
print(x_text_feats.shape)
print("Dimension of description features:")
print(x_description_feats.shape)
print("Dimension of hashtag features:")
print(x_hashtags_feats.shape)

print("Start training and predict...")
kf = KFold(n_splits=10)
fout = open(os.path.join(data_dir, 'integrated_fitted_classes.txt'), 'w')
text_pred = []
description_pred = []
hashtag_pred = []
f = lambda str: str.replace('\n', '')

for train, test in kf.split(x_text_feats):
	model = MultinomialNB().fit(x_text_feats[train], y[train])
	predicts = model.predict(x_text_feats[test])
	text_pred += predicts.tolist()
	text_pred = [f(z) for z in text_pred]

for train, test in kf.split(x_description_feats):
	model = MultinomialNB().fit(x_description_feats[train], y[train])
	predicts = model.predict(x_description_feats[test])
	description_pred += predicts.tolist()
	description_pred = [f(z) for z in description_pred]

for train, test in kf.split(x_hashtags_feats):
	model = MultinomialNB().fit(x_hashtags_feats[train], y_new[train])
	predicts = model.predict(x_hashtags_feats[test])
	hashtag_pred += predicts.tolist()
	hashtag_pred = [f(z) for z in hashtag_pred]

text_pred_new = [pred for k, pred in enumerate(text_pred) if k not in void_ids]
description_pred_new = [pred for k, pred in enumerate(description_pred) if k not in void_ids]
y_new = np.array([f(z) for z in y_new])

with open(os.path.join(data_dir, 'integrated_fitted_classes.txt'), 'w') as fout:
	fout.write('TEXT\tDESCRIPTION\tHASHTAGS\tLABEL\n')
	for counter in range(len(y_new)):
		fout.write('\t'.join([text_pred[counter], description_pred[counter], hashtag_pred[counter], y_new[counter]]) + '\n')
print('The classes of %d tweets are predicted.' %len(y_new))
fout.close()


## Blend the prediction from three classifiers
x_blends = []
for m in range(len(y_new)):
	x_blend = [int(text_pred[m]), int(description_pred[m]), int(hashtag_pred[m])]
	x_blends.append(x_blend)

x_blend_feats = scipy.sparse.csr_matrix(x_blends)
avg_p = 0
avg_r = 0
for train, test in kf.split(x_blend_feats):
	clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
	rf = clf.fit(x_blend_feats[train], y_new[train])
	predicts = rf.predict(x_blend_feats[test])
	# model = MultinomialNB().fit(x_blend_feats[train], y_new[train])
	# predicts = model.predict(x_blend_feats[test])
	print(classification_report(y_new[test],predicts))
	avg_p += precision_score(y_new[test],predicts, average='macro')
	avg_r += recall_score(y_new[test],predicts, average='macro')

print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))
