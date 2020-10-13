import re
import string
import pandas as pd
import ftfy
import nltk
#spacy
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

# plot
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


def clean(doc):
    '''
    fuction that removes punct,stopwords,tokenize and creating Pos Tags
    for every token.
    Token= every word individualy.
    '''
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    stop_free = ftfy.fix_text(stop_free)
    # stop_free=ftfy.fix_text(stop_free)
    # txt_clean=""
    # for word in stop_free:
    #     for j in enumerate(startlist):
    #         txt_clean = "".join(word for word in stop_free if not word.startswith('@','pic.twitter.com',"#",))
    txt_clean = ''.join(word for word in stop_free if not word.startswith(starttuple))
    punc_free = ''.join(ch for ch in txt_clean if ch not in exclude)
    normalized = ''.join(word for word in punc_free if ps.stem(word))
    processed = re.sub(r"\d+", "", normalized)
    POStokens = nltk.word_tokenize(processed)
    #POStokens = [word for word in nltk.pos_tag(nltk.word_tokenize(processed))]
    return POStokens


def strip_emoji(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    return RE_EMOJI.sub(r'', text)

### clean with sapcy

def Sclean(dw):
    let=nlp(dw).lower()
    mini_clean = [i for i in let if not let.is_stop and not let.is_punct]
    stop_free = " ".join([i for i in dw.lower().split() if nlp.vocab[i].is_stop == False])
    let= nlp(stop_free)
    punc_free = " ".join(token.text for token in let if not token.is_punct)
    let = nlp(punc_free)
    edit_lemma = " ".join([token.lemma_ for token in let])
    include_pos = " ".join([token.pos_ for token in let ])
    processed = re.sub(r"\d+", "", edit_lemma)
    return processed

#defenition of the variable need for the clean function
starttuple = ('@', 'pic.twitter.com', "#", "http", "bit.ly")
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
ps = PorterStemmer()



'''
here we load the files and combine different Sheets and columns to 
have the out come that we need from the the excel files that we have
'''
data = pd.ExcelFile(r'data/text only.xlsx')
    
data1 = pd.ExcelFile(r'data/Twitter_Data.xlsx')

'''
Cobination Excel sheets of datasets for better efficiency nad dropping cols that we cant find any use
'''
insta = pd.read_excel(data, sheet_name=None)
insta = pd.concat(insta, axis=1,sort=True, ignore_index=True)
insta_placebo= pd.read_excel(data1, sheet_name=None)
insta_placebo = pd.concat(insta_placebo,sort=False,ignore_index=True)

insta.dropna(axis=1,how="all",inplace=True)



#merging assets of 2 datasets for more data for text analysis
for i, y in enumerate(insta_placebo.Language):
        if y  != "en":
            insta_placebo.drop(i, inplace=True)

#this is a fuction with a specific ult
def check(doc,kk):
    lists =[]
    if doc == "Retweet":
        lists.append(kk)
    return lists



#new dataset with all the text from caption
#we have a corpus with 10k documents
df=insta[2].append(insta[9],ignore_index=True)
df=df.append(insta_placebo.Tweet,ignore_index=True)


df.dropna(inplace=True,how="all")

df=df.apply(lambda x: strip_emoji(str(x)))

prehash=df.append(insta[5],ignore_index=True)
prehash.dropna(inplace=True,how="all")

#Code that takes and stores in dictionaries Mentions and hashtag
tag_dict = {}
mention_dict = {}
for i in prehash:

    tweet = str(i).lower()
    tweet_tokenized = tweet.split()
    for word in tweet_tokenized:
        # Hashtags - tokenize and build dict of tag counts
        if (word[0:1] == '#' and len(word) > 1):
            key = word.translate(string.punctuation)
            if key in tag_dict:
                tag_dict[key] += 1
            else:
                tag_dict[key] = 1
        # Mentions - tokenize and build dict of mention counts
        if (word[0:1] == '@' and len(word) > 1):
            key = word.translate(string.punctuation)
            if key in mention_dict:
                mention_dict[key] += 1
            else:
                mention_dict[key] = 1



df=df.apply(lambda x: clean(str(x)))

hashtag=pd.DataFrame.from_dict(tag_dict,orient="index")
hashtag.sort_values(by=[0],ascending=False,inplace=True)

'''
we create a dictionary and requirements for the Lda function.
so we specify in the model building on how the algorithm will work
and we print the mesurments that show us the success rate of the training
'''
id2word = corpora.Dictionary(df)
# Create Corpus
texts = df
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# model building
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=150,
                                            update_every=1,
                                            chunksize=150,
                                            passes=20,
                                            alpha='auto',
                                            per_word_topics=True)

print('\nPerplexity: ', lda_model.log_perplexity(corpus))

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


'''
here we are produce visuals of the analysis and we also export it 
on html file
'''

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word,mds='mmds')
#or display
pyLDAvis.show(vis)

#pyLDAvis.save_html(vis,'/Users/DSS/Desktop/ClimateTest/Thepopecode/bestvis.html')
#create ecel file
#hashtag.to_excel("OfficialFreqHash.xlsx")  