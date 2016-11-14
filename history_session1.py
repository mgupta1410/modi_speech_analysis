######## Replace contractions. Remove apostrophes ######################
import pickle
speeches = pickle.load(open('speeches.pickle','r'))
contractions = pickle.load(open('contractions.pickle','r'))

def replace_map_string(map_, str_):
    for key_ in map_:
        str_ = str_.replace(key_, map_[key_])
    return str_

for i in range(175):
    speeches['title'][i] = replace_map_string(contractions, speeches['title'][i])
    speeches['text'][i] = replace_map_string(contractions, speeches['text'][i])

for i in range(175):
    speeches['title'][i] = speeches['title'][i].replace("'", "")
    speeches['text'][i] = speeches['text'][i].replace("'", "")

pickle.dump(speeches, open('speeches_apostrope.pickle', 'w' ))

############# tokenizing and joining###########################
from nltk.tokenize import word_tokenize
speeches_tokenized = speeches
for i in range(175):
    speeches_tokenized['title'][i] = ' '.join(word_tokenize(speeches_tokenized['title'][i]))
    speeches_tokenized['text'][i]  = ' '.join(word_tokenize(speeches_tokenized['text'][i]))
speeches_tokenized['text'][5]
pickle.dump(speeches_tokenized, open('speeches_tokenized.pickle', 'w' ))

############# lower casing, "mr."->"mr", replacing "a.b" -> "a . b", removing back ticks ###########################
speeches_tokenized_lower = speeches_tokenized
for i in range(175):
    speeches_tokenized_lower['title'][i] = speeches_tokenized['title'][i].lower()
    speeches_tokenized_lower['text'][i]  = speeches_tokenized['text'][i].lower()

for i in range(175):
    speeches_tokenized_lower['title'][i] = speeches_tokenized['title'][i].replace("mr. ", "mr ")
    speeches_tokenized_lower['text'][i]  = speeches_tokenized['text'][i].replace("mr." , "mr")
speeches_tokenized_lower['text'][4]
import re

def replace_dot(str):
    return re.sub(r'([a-z])\.([a-z])', r'\1 . \2', str)

for i in range(175):
    speeches_tokenized_lower['title'][i] = replace_dot(speeches_tokenized['title'][i])
    speeches_tokenized_lower['text'][i]  = replace_dot(speeches_tokenized_lower['text'][i])

for i in range(175):
    speeches_tokenized_lower['title'][i] = speeches_tokenized['title'][i].replace("`", "")
    speeches_tokenized_lower['text'][i]  = speeches_tokenized['text'][i].replace("`" , "")

pickle.dump(speeches_tokenized_lower, open('speeches_tokenized_lower.pickle', 'w' ))
############************************************************************###########
##four kinds of frequency lists from speeches_tokenized_lower -
# 1) with stopwords, 2) stopwords cleaned, 3) extensive stopwords cleaned, 4) extensive + lemmatized

import pickle
speeches_tokenized_lower = pickle.load(open('speeches_tokenized_lower.pickle', 'r' ));

stop_nonstop_words_freq = {}

punctuation ='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

for i in range(175):
    lst = speeches_tokenized_lower['text'][i].split()
    for token in lst:
        if token not in punctuation:
            if token in stop_nonstop_words_freq:
                stop_nonstop_words_freq[token] += 1
            else:
                stop_nonstop_words_freq[token] = 0

stop_nonstop_words_freq_lst = stop_nonstop_words_freq.items()
nonstop_words_freq_lst = []

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

nonstop_words_freq_lst = []
for tup in stop_nonstop_words_freq_lst:
    word = tup[0]
    if word not in stopwords:
        nonstop_words_freq_lst.append(tup)

stopwords_extensive = pickle.load(open('stopwords_extensive.pickle', 'r'))

ex_nonstop_words_freq_lst = []
for tup in stop_nonstop_words_freq_lst:
    word = tup[0]
    if word not in stopwords_extensive:
        ex_nonstop_words_freq_lst.append(tup)
ex_nonstop_words_freq_lst.sort(key=lambda tup: tup[1], reverse=True)

#lemmatized
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

ex_lem_nonstop_words_freq = {}
for tup in ex_nonstop_words_freq_lst:
    word = tup[0]
    num = tup[1]
    lemma = str(wordnet_lemmatizer.lemmatize(word))
    if lemma in ex_lem_nonstop_words_freq:
        ex_lem_nonstop_words_freq[lemma] += num
    else:
        ex_lem_nonstop_words_freq[lemma] = num

ex_lem_nonstop_words_freq_lst = ex_lem_nonstop_words_freq.items()
ex_lem_nonstop_words_freq_lst = sorted(ex_lem_nonstop_words_freq_lst, key=lambda a:a[1], reverse=True)
pickle.dump(ex_lem_nonstop_words_freq_lst, open('ex_lem_nonstop_words_freq_lst.pickle', 'w'))
##########creating bar chart#############################
import matplotlib.pyplot as plt
top_words = ex_lem_nonstop_words_freq_lst[0:60]
top_words_freq = []
top_words_name = []
for tup in top_words:
    top_words_name.append(tup[1])
    top_words_freq.append(tup[1])

index = np.arange(60)
bar_width = 0.8
opacity = 0.6
rects = plt.bar(index, top_words_freq, bar_width, alpha=opacity)
plt.ylabel('frequency')
plt.title('Frequency of key words')
plt.subplots_adjust(bottom=0.2)
plt.xticks(index + bar_width, top_words_names, rotation='vertical')
plt.xlabel('words')

#lemmatized wordcloud
all_text_lem_bag = ""
for tup in ex_lem_nonstop_words_freq_lst:
    num = tup[1]; word = tup[0]
    for i in range(num):
        all_text_lem_bag += " " + word

import wordcloud
from wordcloud import WordCloud

import matplotlib.pyplot as plt
# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(max_font_size=40, relative_scaling=0.5).generate(all_text_lem_bag)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


##############################################################################
#getting sentences#
speeches_tokenized_lower = pickle.load(open('speeches_tokenized_lower.pickle', 'r' ))
import re
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

speeches_sentences = speeches_tokenized_lower

for i in range(175):
    speeches_sentences['text'][i] = split_into_sentences(speeches_tokenized_lower['text'][i])

pickle.dump(speeches_sentences, open('speeches_sentences.pickle','w'))

#######LDA on sentences as documents########
import pickle
stopwords_extensive = pickle.load(open('stopwords_extensive.pickle', 'r'))
speeches_sentences = pickle.load(open('speeches_sentences.pickle','r'))
speeches_sentences_text = speeches_sentences['text']

#sentences in plain text
sentences_text_plain = []

import re
punctuation ='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
regex = re.compile('[%s]' % re.escape(punctuation))

def test_re(s):
    return ' '.join(regex.sub('', s).split())

for i in range(175):
    lst = speeches_sentences_text[i]
    for sent in lst:
        sentences_text_plain.append(test_re(sent))

#removal of extensive stopwords and lemmatization
sentences_text_plain_ex_lem = []
stopwords_extensive = set(pickle.load(open('stopwords_extensive.pickle', 'r')))
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

for sent in sentences_text_plain:
    new_sent = ""
    for word in sent.split():
        if word not in stopwords_extensive:
            new_sent += " " + str(wordnet_lemmatizer.lemmatize(word))
    sentences_text_plain_ex_lem.append(new_sent)

texts = []
for sent in sentences_text_plain_ex_lem:
    texts.append(sent.split())

from gensim import corpora, models
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=10)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=50)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=100)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=40, id2word = dictionary, passes=200)


#######LDA on speeches as documents########
import pickle
stopwords_extensive = pickle.load(open('stopwords_extensive.pickle', 'r'))
speeches_tokenized_lower = pickle.load(open('speeches_tokenized_lower.pickle','r'))
speeches_tokenized_lower_text = speeches_tokenized_lower['text']

documents_text_plain = []

import re

def test_re(s):
    punctuation ='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    regex = re.compile('[%s]' % re.escape(punctuation))
    return ' '.join(regex.sub('', s).split())

for i in range(175):
    documents_text_plain.append(test_re(speeches_tokenized_lower_text[i]))

#removal of extensive stopwords and lemmatization
documents_text_plain_ex_lem = []
stopwords_extensive = set(pickle.load(open('stopwords_extensive.pickle', 'r')))
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

for doc in documents_text_plain:
    new_doc = ""
    for word in doc.split():
        if word not in stopwords_extensive:
            new_doc += " " + str(wordnet_lemmatizer.lemmatize(word))
    documents_text_plain_ex_lem.append(new_doc)

texts = []
for sent in documents_text_plain_ex_lem:
    texts.append(sent.split()) #pickled to documents_words.pickle

import gensim
from gensim import corpora, models
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=100)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=200) #lda_doc_200

i = 6
while i > 0:
    if(i != 4):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i*5, id2word = dictionary, passes=200)
        prepared = pg.prepare(ldamodel, corpus, dictionary)
        pyLDAvis.save_html(prepared, open('lda_doc_topics' + str(i*5) + '.html', 'w'))
    i -= 1

i = 6
while i > 0:
    if(i != 4):
        ldamodel = pickle.load(open('ldamodel_doc_topics' + str(i*5), 'r') )
        prepared = pg.prepare(ldamodel, corpus, dictionary)
        pyLDAvis.save_html(prepared, open('lda_doc_topics' + str(i*5) + '.html', 'w'))
    i -= 1
ldamodel = pickle.load(open('ldamodel_doc_topics30', 'r') )
pyLDAvis.save_html(prepared, open('lda_doc_topics30.html', 'w'))

############################LDA VISUALIZATION##########################################
import pyLDAvis
from pyLDAvis import gensim as pg
prepared = pg.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(pyLDAvis.display(prepared), open('lda.html', 'w'))# not correct mostly 


#######################################################################################
#GET ALL DATES#
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

datesu = []
titles = []
def get_titles():
    url = 'http://www.narendramodi.in/category/text-speeches'
    driver = webdriver.Chrome()
    driver.get(url)
    # First scroll to the end of the table by sending Page Down keypresses to
    # the browser window.
    num = 350
    #while driver.find_element_by_class_name('pwdBy').text.strip() != 'July 23, 2012':
    elem = driver.find_element_by_tag_name('a')
    while num > 0 and elem is not None:
        # Find the first element on the page, so we can scroll down using the
        # element object's send_keys() method
        if elem is None:
            break;
        elem.send_keys(Keys.PAGE_DOWN)
        print num
        num -= 1
        elem = driver.find_element_by_tag_name('a')

    # Once the whole table has loaded, grab all the visible links.
    classes = driver.find_elements_by_class_name('speechesItemLink')
    for each in classes:
        titles.append(each.text)
    driver.quit()
########################################################################
#barchart for speeches in a month# Jun 2014 - Aug 2016
from datetime import datetime
for date in datesu:
    dates.append(str(date).replace(',' , '')) #pickled in dates.pickle

month_freq_map = {}
for each in dates:
    key = datetime.strptime(each,'%B %d %Y').strftime('%m#%y'); print key
    if key in month_freq_map:
        month_freq_map[key] += 1
    else:
        month_freq_map[key] = 0

month_freq = month_freq_map.items()
month_freq.sort(key=lambda tup: tup[1], reverse=True)

import matplotlib.pyplot as plt

month_freq_filt = []
for val in month_freq:
    month = long(val[0].split('#')[0])
    year  = long(val[0].split('#')[1])
    if(not (year < 14L) and not (month < 6L and year == 14L) and not (month > 8 and year ==16)):
        month_freq_filt.append((val[0].split('#')[1] + '#' + val[0].split('#')[0], val[1]))

month_freq_filt.sort()
month_freq_filt_lst = []
month_names = []

for tup in month_freq_filt:
    month_freq_filt_lst.append(tup[1])
    month_names.append(datetime.strptime(tup[0],'%y#%m').strftime('%b \'%y'))

import numpy as np
index = np.arange(25)
bar_width = 0.5
opacity = 0.4
plt.ylabel('No of speeches')
plt.title('Frequency of speeches')
plt.xlabel('Months')
plt.subplots_adjust(bottom=0.2)
plt.xticks(index + bar_width*1.0/2, month_names, rotation='vertical')
plt.plot(index + bar_width*1.0/2, month_freq_filt_lst)
plt.bar(index, month_freq_filt_lst, bar_width, alpha=opacity, color='r')

#####################################Lexical Diversity####################################################
texts = pickle.load(open('documents_words.pickle', 'r'));
ld = []
for i in range(175):
    ld.append( len(set(texts[i])) / (len(texts[i]) * 1.0) )

ld_date_place_title = []
speeches = pickle.load(open('speeches_tokenized_lower.pickle', 'r'))
for i in range(175):
    ld_date_place_title.append((ld[i], speeches['date'][i], speeches['place'][i], speeches['title'][i] ))

ld_india_inter = []
for i in range(175):
    place = 1
    if(ld_date_place_title[i][2].strip().lower() == 'india'):
        place = 0
    ld_india_inter.append((ld_date_place_title[i][0], place))

#stacked bar chart - national vs international lexical Diversity

indian = [0, 0, 0, 0]
inter = [0, 0, 0, 0]
for num in range(4):
    init = 0.4 + num*0.1
    exc = init + 0.1
    for i in range(175):
        if(ld_india_inter[i][0] >= init and ld_india_inter[i][0] < exc):
            inter[num] += ld_india_inter[i][1]
            if(ld_india_inter[i][1] == 0):
                indian[num] += 1

import matplotlib.pyplot as plt
import numpy as np
f, ax1 = plt.subplots(1, figsize=(10,5))
bar_width = 0.75
bar_l = [i+1 for i in range(len(indian))]
tick_pos = [i+(bar_width/2) for i in bar_l]
plt.xticks(tick_pos, ['0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8'])
ax1.bar(bar_l, indian, width=bar_width, label='India', alpha=0.5, color='g')
ax1.bar(bar_l, inter, width=bar_width, label='International', alpha=0.5, color='b', bottom=indian)
plt.legend(loc='upper left')
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim([0, 100])
