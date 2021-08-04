# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:21:32 2021

@author: cocogne
"""
# select *
# from Posts
# where ViewCount>10000 and  (YEAR(CreationDate))>2020 and (YEAR(CreationDate))<2022
# pour la derniere requete


import pandas as pd
import os
from collections import defaultdict
import nltk
import re
from nltk.stem.snowball import EnglishStemmer
import numpy as np

path_to_data = os.path.join('c:\\','openclassrooms','projet5','projet')
data_fname = 'QueryResults.csv'
data = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data=data[['Title','Tags']]

data_fname = 'QueryResults(1).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Tags']]

data=pd.concat([data,data2])

data_fname = 'QueryResults(2).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Tags']]

data=pd.concat([data,data2])

data_fname = 'QueryResults(3).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Tags']]

data=pd.concat([data,data2])


data_dict=data.set_index('Title').to_dict('index')
data_dict_questions_tags = defaultdict(set)
for k in data_dict:
    tags=data_dict[k]['Tags']
    data_dict_questions_tags[k]=tags

#nombre de tags
base_tags = defaultdict(set)
for k,v in data_dict_questions_tags.items():
    base_tags[v].add(k)

tags = { k:v for k,v in base_tags.items() if len(v) > 2 }
print('{} liste de tags'.format(len(tags)))



#nombre de mots par tag

tokenizer = nltk.RegexpTokenizer(r'\w+')
corpora = defaultdict(list)
# Création d'un corpus de tokens par artiste
for tag,question_id in tags.items():
    for question in question_id:
        corpora[tag] += tokenizer.tokenize(
                                question.lower()
                            )

stats, freq = dict(), dict()

for k, v in corpora.items():
    freq[k] = fq = nltk.FreqDist(v)
    stats[k] = {'total': len(v)} 

# Récupération des comptages
df = pd.DataFrame.from_dict(stats, orient='index')

# Affichage des fréquences
df=df.sort_values(by='total', ascending=False)
df[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots')


#nombre de mots uniques par tag
corpora = defaultdict(list)
for tag,question_id in tags.items():
    for question in question_id:
        corpora[tag] += tokenizer.tokenize(
                                question.lower()
                            )
    
stats_unique, freq_unique = dict(), dict()

for k, v in corpora.items():
    freq_unique[k] = fq = nltk.FreqDist(v)
    stats_unique[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique = pd.DataFrame.from_dict(stats_unique, orient='index')

# Affichage des fréquences
df_unique=df_unique.sort_values(by='total', ascending=False)
df_unique[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')



#gestion des stopwords

# Premièrement, on récupère la fréquence totale de chaque mot sur tout le corpus d'artistes
freq_totale = nltk.Counter()
for k, v in corpora.items():
    freq_totale += freq[k]

# Deuxièmement on décide manière un peu arbitraire du nombre de mots les plus fréquents à supprimer. 
#On pourrait afficher un graphe d'évolution du nombre de mots pour se rendre compte et avoir une meilleure heuristique. 

#most_freq=pd.Series(' '.join(df.text).lower().split()).value_counts()[:100]
most_freq = list(zip(*freq_totale.most_common(100)))[0]
print(most_freq)
most_freq = list(zip(*freq_totale.most_common(9)))[0]

nltk.download('stopwords')
# On créé notre set de stopwords final qui cumule ainsi les 100 mots les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut présent dans la librairie NLTK
sw = set()
plus=(' ','  ','   ','    ','      ','        ','         ')
sw.update(plus)
sw.update(most_freq)
sw.update(tuple(nltk.corpus.stopwords.words('english')))


corpora_sans_stopwords = defaultdict(list)
for tag, question_id in tags.items():
    for question in question_id:
        tokens = tokenizer.tokenize(question.lower())
        corpora_sans_stopwords[tag] += [w for w in tokens if (not w in list(sw)) ]
stats_sans_stopwords, freq_sans_stopwords = dict(), dict()

for k, v in corpora_sans_stopwords.items():
    freq_sans_stopwords[k] = fq = nltk.FreqDist(v)
    stats_sans_stopwords[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique_sans_stopwords = pd.DataFrame.from_dict(stats_sans_stopwords, orient='index')

# Affichage des fréquences
df_unique_sans_stopwords=df_unique_sans_stopwords.sort_values(by='total', ascending=False)
df_unique_sans_stopwords[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')


# ajout de la lemmetisation
#import sys
#from unicodedata import category
#codepoints = range(sys.maxunicode + 1)
#ponctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")}
#sw.update(ponctuation)

import spacy
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")
print(lemmatizer.mode)

corpora_sans_stopwords_lemme = defaultdict(list)
for tag, question_id in tags.items():
    for question in question_id:
        question=re.sub(r'[^\w\s\']',' ',question)
        doc=nlp(question.lower())
        tokens=[token.lemma_ for token in doc]
        corpora_sans_stopwords_lemme[tag] += [w for w in tokens if ( not w in list(sw))]

stats_sans_stopwords_lemme, freq_sans_stopwords_lemme = dict(), dict()

for k, v in corpora_sans_stopwords_lemme.items():
    freq_sans_stopwords_lemme[k] = fq = nltk.FreqDist(v)
    stats_sans_stopwords_lemme[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique_sans_stopwords_lemme = pd.DataFrame.from_dict(stats_sans_stopwords_lemme, orient='index')

df_unique_sans_stopwords_lemme=df_unique_sans_stopwords_lemme.sort_values(by='total', ascending=False)
df_unique_sans_stopwords_lemme[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')




#ajout de la racinisation spacy et suppression des accents
import unicodedata

suffixes = list(nlp.Defaults.suffixes)
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

prefixes = list(nlp.Defaults.prefixes)
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search

corpora_sans_stopwords_lemme_stem_spacy = defaultdict(list)
for tag, question_id in tags.items():
    for question in question_id:
        question=str(unicodedata.normalize('NFD', question).encode('ascii', 'ignore'))[2:-1] 
        question=re.sub(r'[^\w\s\']',' ',question)
        doc=nlp(question.lower())
        tokens=[token.lemma_ for token in doc]
        corpora_sans_stopwords_lemme_stem_spacy[tag] += [w for w in tokens if ( not w in list(sw))]

stats_sans_stopwords_lemme_stem_spacy, freq_sans_stopwords_lemme_stem_spacy = dict(), dict()

for k, v in corpora_sans_stopwords_lemme_stem_spacy.items():
    freq_sans_stopwords_lemme_stem_spacy[k] = fq = nltk.FreqDist(v)
    stats_sans_stopwords_lemme_stem_spacy[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique_sans_stopwords_lemme_stem_spacy = pd.DataFrame.from_dict(stats_sans_stopwords_lemme_stem_spacy, orient='index')

# Affichage des fréquences
df_unique_sans_stopwords_lemme_stem_spacy=df_unique_sans_stopwords_lemme_stem_spacy.sort_values(by='total', ascending=False)
df_unique_sans_stopwords_lemme_stem_spacy[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')



#ajout de la racinisation nltk 

stemmer = EnglishStemmer()


suffixes = list(nlp.Defaults.suffixes)
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

prefixes = list(nlp.Defaults.prefixes)
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search

corpora_sans_stopwords_lemme_stem = defaultdict(list)
for tag, question_id in tags.items():
    for question in question_id:
        question=str(unicodedata.normalize('NFD', question).encode('ascii', 'ignore'))[2:-1] 
        question=re.sub(r'[^\w\s\']',' ',question)
        doc=nlp(question.lower())
        tokens=[token.lemma_ for token in doc]
        corpora_sans_stopwords_lemme_stem[tag] += [stemmer.stem(w) for w in tokens if ( not w in list(sw))]

stats_sans_stopwords_lemme_stem, freq_sans_stopwords_lemme_stem = dict(), dict()

for k, v in corpora_sans_stopwords_lemme_stem.items():
    freq_sans_stopwords_lemme_stem[k] = fq = nltk.FreqDist(v)
    stats_sans_stopwords_lemme_stem[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique_sans_stopwords_lemme_stem = pd.DataFrame.from_dict(stats_sans_stopwords_lemme_stem, orient='index')

# Affichage des fréquences
df_unique_sans_stopwords_lemme_stem=df_unique_sans_stopwords_lemme_stem.sort_values(by='total', ascending=False)
df_unique_sans_stopwords_lemme_stem[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')




#correction de language avec speller from autocorrect
# =============================================================================
# from autocorrect import Speller
# import time
# t1=time.time()
# 
# #Enlever le Fast=True
# 
# spell = Speller()
# corpora_sans_stopwords_lemme_stem_speller = defaultdict(list)
# for tag, question_id in tags.items():
#     for question in question_id:
#         question=str(unicodedata.normalize('NFD', question).encode('ascii', 'ignore'))[2:-1] 
#         question=re.sub(r'[^\w\s\']',' ',question)
#         question=spell(question.lower())
#         doc=nlp(question)
#         tokens=[token.lemma_ for token in doc]
#         corpora_sans_stopwords_lemme_stem_speller[tag] += [stemmer.stem(w) for w in tokens if (not w in list(sw)) ]
# 
# t2=time.time()
# print ("speller prend {} min à tourner".format((t2-t1)/60))
# stats_sans_stopwords_lemme_stem_speller, freq_sans_stopwords_lemme_stem_speller = dict(), dict()
# 
# for k, v in corpora_sans_stopwords_lemme_stem_speller.items():
#     freq_sans_stopwords_lemme_stem_speller[k] = fq = nltk.FreqDist(v)
#     stats_sans_stopwords_lemme_stem_speller[k] = {'total': len(v), 'unique': len(fq.keys())}
# 
# df_unique_sans_stopwords_lemme_stem_speller = pd.DataFrame.from_dict(stats_sans_stopwords_lemme_stem_speller, orient='index')
# 
# # Affichage des fréquences
# df_unique_sans_stopwords_lemme_stem_speller=df_unique_sans_stopwords_lemme_stem_speller.sort_values(by='total', ascending=False)
# df_unique_sans_stopwords_lemme_stem_speller[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')
# 

 
# =============================================================================
#correction de language avec speller from autocorrect sans racinisation nltk
from autocorrect import Speller
import time
t1=time.time()

#Enlever le Fast=True

spell = Speller()
corpora_sans_stopwords_lemme_stem_sans_nltk_speller = defaultdict(list)
for tag, question_id in tags.items():
    for question in question_id:
        question=str(unicodedata.normalize('NFD', question).encode('ascii', 'ignore'))[2:-1] 
        question=re.sub(r'[^\w\s\']',' ',question)
        question=spell(question.lower())
        doc=nlp(question)
        tokens=[token.lemma_ for token in doc]
        corpora_sans_stopwords_lemme_stem_sans_nltk_speller[tag] += [w for w in tokens if (not w in list(sw)) ]

t2=time.time()
print ("speller prend {} min à tourner".format((t2-t1)/60))
stats_sans_stopwords_lemme_stem_sans_nltk_speller, freq_sans_stopwords_lemme_stem_sans_nltk_speller = dict(), dict()

for k, v in corpora_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    freq_sans_stopwords_lemme_stem_sans_nltk_speller[k] = fq = nltk.FreqDist(v)
    stats_sans_stopwords_lemme_stem_sans_nltk_speller[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique_sans_stopwords_lemme_stem_sans_nltk_speller = pd.DataFrame.from_dict(stats_sans_stopwords_lemme_stem_sans_nltk_speller, orient='index')

# Affichage des fréquences
df_unique_sans_stopwords_lemme_stem_sans_nltk_speller=df_unique_sans_stopwords_lemme_stem_sans_nltk_speller.sort_values(by='total', ascending=False)
df_unique_sans_stopwords_lemme_stem_sans_nltk_speller[0:49].plot(kind='bar', color="#f56900", title='Top 50 des tags par nombre de mots uniques')



list_question=[]
list_tag=[]
for tag, question_id in corpora_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    question=" ".join(corpora_sans_stopwords_lemme_stem_sans_nltk_speller[tag])
    list_question.append(question)
    list_tag.append(tag)
    
from sklearn.feature_extraction.text import CountVectorizer
#max_features=nombre de mots que l'on va garder dans le vocabulaire
tf_vectorizer=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
tf=tf_vectorizer.fit_transform(list_question)



def save_topics(model, feature_names, no_top_words,n_topic):
    data_topic['topic_feature_lda_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(model.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic['topic_feature_lda_'+str(n_topic)][topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        
    
from sklearn.decomposition import LatentDirichletAllocation
#a tester hyperparametres nbr de sujets
n_topics = np.arange(10,101,5)
data_topic = pd.DataFrame(index=np.arange(100))
for n_topic in n_topics:
    # Créer le modèle LDA
    lda = LatentDirichletAllocation(
            n_components=n_topic, 
            max_iter=5, 
            learning_method='online', 
            learning_offset=50.,
            random_state=0)
    
    # Fitter sur les données
    lda.fit(tf)
    save_topics(lda, tf_vectorizer.get_feature_names(), 20, n_topic)  


#search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9],'random_state':0}
search_params = {'n_components': np.arange(10,101,5), 
                 'learning_decay': [.5, .7, .9],
                 'random_state':0,
                 'learning_method':'online'}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))