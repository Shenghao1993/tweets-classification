import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score

features_dir = './features'
data_dir = './data'

print("Loading hashtags...")
with open(os.path.join(features_dir, 'hashtags_processed.txt'), 'r') as f:
	x = f.readlines()
	void_ids = [i for i, tag in enumerate(x) if tag == '\n']
	x_new = [tag for j, tag in enumerate(x) if j not in void_ids]

with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
	y = np.array(f.readlines())
	y_new = np.array([label for k, label in enumerate(y) if k not in void_ids])

print("Extract features...")
x_feats = TfidfVectorizer().fit_transform(x_new)
print(x_feats.shape)

print("Start training and predict...")
kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
# fout = open(os.path.join(data_dir, 'hashtag_fitted_classes.txt'), 'w')
hashtag_pred = ''
for train, test in kf.split(x_feats):
	model = MultinomialNB().fit(x_feats[train], y_new[train])
	# hidden_layer_sizes=(no_of_hidden_units, no_of_hidden_layers)
	# clf = MLPClassifier(solver='adam', alpha=1e-5,
	#                     hidden_layer_sizes=(100, 50, 20), random_state=1)
	# model = clf.fit(x_feats[train], y_new[train])
	predicts = model.predict(x_feats[test])
	hashtag_pred += ''.join(predicts)
	print(classification_report(y_new[test],predicts))
	avg_p += precision_score(y_new[test],predicts, average='macro')
	avg_r += recall_score(y_new[test],predicts, average='macro')
# fout.write('%s' %hashtag_pred)
# fout.close()
print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))