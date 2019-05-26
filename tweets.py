import codecs
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import sys
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
import emoji

with codecs.open('onlineHarassmentDataset.tdf', 'r', encoding='utf-8',errors='ignore') as f:
		data = f.readlines()


text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

seg_tw = Segmenter(corpus="twitter")
sp = SpellCorrector(corpus="twitter") 
f1 = open('tokenized_tweets_golbeck.txt', 'w')
c=1
for line in data:
	a = line.strip().split('\t')
	if len(a)>=3:
		b = a[2]
		c = a[1]
		b = b.split()
		for i in range(len(b)):
			if b[i].startswith('http'):
				b[i] = '<url>'
		b = ' '.join(b)
		a = text_processor.pre_process_doc(b)
		for i in range(len(a)):
			if a[i].isalpha():
				a[i] = seg_tw.segment(sp.correct(a[i]))
		a = ' '.join(a)
		f1.write(a+' '+c+'\n')


