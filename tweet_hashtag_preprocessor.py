import os, re, nltk
import simplejson as json
from nltk.corpus import stopwords

def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>',re.S)
    return html_prog.sub('', str)

def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)

def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)

def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)

def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)

def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)

def replace_emoticon(emoticon_dict, str):
    for k, v in emoticon_dict.items():
        str = str.replace(k, v)
    return str

def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]','',current_tweet)

def pre_process(str, porter):
    # do not change the preprocessing order only if you know what you're doing 
    str = str.lower()
    str = rm_url(str)        
    str = rm_at_user(str)        
    str = rm_repeat_chars(str) 
    str = rm_hashtag_symbol(str)       
    str = rm_time(str)        
    str = rm_punctuation(str)
        
    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass
        
    return str



if __name__ == "__main__":
	data_dir = './data'
	tweet_source_file = 'samples.txt'

	porter = nltk.PorterStemmer()
	stops = set(stopwords.words('english'))

	## Load and process sample tweets
	print('start loading and process samples...')
	hashtags_stat = {} # record statistics of the df and tf for each hashtag; Form: {tag:[tf, df, tweet index]}
	hashtags = []
	cnt = 0
	with open(os.path.join(data_dir, tweet_source_file)) as f:
		for i, line in enumerate(f):
			postprocess_hashtag_list = []
			tweet_obj = json.loads(line.strip(), encoding='utf-8')
			hashtag_list = tweet_obj['entities']['hashtags']
			no_of_hashtags = len(hashtag_list)
			hashtag_text_list = []
			if no_of_hashtags == 0:
				# joined_postprocess_tags = ''
				joined_postprocess_tags = 'void'
				# hashtags.append(joined_postprocess_tags)
			else:
				for j in range(no_of_hashtags):
					hashtag_text_list.append(hashtag_list[j]['text'])
				joined_tags = ' '.join(hashtag_text_list)
				tags = pre_process(joined_tags, porter)
				for tag in tags:
					if tag not in stops:
						postprocess_hashtag_list.append(tag)
						if tag in hashtags_stat.keys():
							hashtags_stat[tag][0] += 1
							if i != hashtags_stat[tag][2]:
								hashtags_stat[tag][1] += 1
								hashtags_stat[tag][2] = i
						else:
							hashtags_stat[tag] = [1,1,i]
				joined_postprocess_tags = ' '.join(postprocess_hashtag_list)
			hashtags.append(joined_postprocess_tags)
		# print(hashtags[:50])

	## Save the statistics of tf and df for each hashtag into file
	print("The number of unique words in data set is %i." %len(hashtags_stat.keys()))
	lowTF_tags = set()
	stats_dir = './stats'
	with open(os.path.join(stats_dir, 'hashtags_statistics.txt'), 'w') as f:
		f.write('TF\tDF\tHASHTAG\n')
		for tag, stat in sorted(hashtags_stat.items(), key=lambda i: i[1], reverse=True):
			f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + tag +  '\n')
			if stat[0]<2:
				lowTF_tags.add(tag)
	print("The number of low frequency words is %d." %len(lowTF_tags))
    
    ## Re-process samples, filter low frequency hashtags...
	features_dir = './features'
	fout = open(os.path.join(features_dir, 'hashtags_processed.txt'), 'w')
	new_hashtags_list = []
	for hashtag in hashtags:
		tags = hashtag.split(' ')
		new = []
		for tag in tags:
			if tag not in lowTF_tags:
				new.append(tag)
		if len(new) == 0:
			new.append('void')
		new_hashtags = ' '.join(new)
		new_hashtags_list.append(new_hashtags)
		fout.write('%s\n' %new_hashtags)
	fout.close()

	print("Preprocessing is completed")