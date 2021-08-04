# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:21:32 2021

@author: cocogne
"""



import pandas as pd
import os
from collections import defaultdict
import nltk
import re
from nltk.stem.snowball import EnglishStemmer
import numpy as np
from bs4 import BeautifulSoup

def list_unique(list_unique):
    """
    Cet fonction prend une liste en entrée et retorune la liste sans doublon
    """
    new_list = [] 
    for i in list_unique : 
        if i not in new_list: 
            new_list.append(i) 
    return new_list


#On ouvre les fichiers csv issu de stackoverflow avec la requete sql suivante 
# select *
# from Posts
# where ViewCount>10000 and  (YEAR(CreationDate))>2020 and (YEAR(CreationDate))<2022
# pour la derniere requete

path_to_data = os.path.join('c:\\','openclassrooms','projet5','projet')
data_fname = 'QueryResults.csv'
data = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data=data[['Title','Body','Tags']]

data_fname = 'QueryResults(1).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Body','Tags']]

data=pd.concat([data,data2])

data_fname = 'QueryResults(2).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Body','Tags']]

data=pd.concat([data,data2])

data_fname = 'QueryResults(3).csv'
data2 = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
data2=data2[['Title','Body','Tags']]

data=pd.concat([data,data2])

#On crée un dictionnaire dict[question][tags],dict[question][Body]
data_dict=data.set_index('Title').to_dict('index')



###########################################################
#                GESTION DES TAGS                         #
###########################################################



#on split la variable tag pour avoir une liste de tag
liste_tags=[]
for k in data_dict:
    data_dict[k]['Tags']=data_dict[k]['Tags'].lower()
    data_dict[k]['Tags']=data_dict[k]['Tags'][1:-1].split("><")
    liste_tags=liste_tags+data_dict[k]['Tags']
 
#On compte les occurences des tags
from collections import Counter
tag_counter=Counter(liste_tags).most_common()

#on crée une liste avec tous les tags
list_tag_unique = list_unique(liste_tags)

#on crée une liste avec les tags qui ont 300 occurences ou plus
list_tag_sup=[]
for element in tag_counter:
    a,b=element
    if b>=300:
        list_tag_sup.append(a)

#on supprime de la liste des tags qui seront traités individuellement par la suite
list_tag_sup.remove("git")
list_tag_sup.remove("java")
list_tag_sup.remove("pip")
list_tag_sup.remove("r")
list_tag_sup.remove(".net-core")
list_tag_sup.remove("sql")

#on crée un dictionnaire de traduction des tags
traduction_tags_plus = defaultdict(set)

#on boucle sur tous les tags possible
for element in list_tag_unique:
    traduction_tags_plus[element]=[]
    
    if (".net-" in element) and (".net-core" not in element) and ("asp.net" not in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append(".net")

    if  (".net-core" in element) and ("asp.net" not in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append(".net-core")
 
    if ("airflow" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("airflow")
    
    if ("amazon" in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("amazon")

    if ("apache" in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("apache")

    if ("asp.net" in element)  and (".net-core" not in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("asp.net")
    
    if ("flask" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("flask")
    
    try:
        indice=element.index("git")
    except ValueError:
        indice=-1
    if ((indice==0) or (" git" in element)) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("git")
    
    try:
        indice=element.index("go-")
    except ValueError:
        indice=-1
    if ((indice==0) and (element not in list_tag_sup)):
        traduction_tags_plus[element].append("go")

    if ("google" in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("google")

    if ("java" in element) and ("javascript" not in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("java")

    if ("js" in element) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("javascript")

    if (("jupyter" in element) or ("jupiter" in element)) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("jupyter")
        
    if ("laravel" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("laravel")

    if ("maven" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("maven")
 
    if ("microsoft" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("microsoft")
 
   
    if ("mysql" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("mysql")
    
    try:
        indice=element.index("ng-")
    except ValueError:
        indice=-1
    if (("ng2-"in element) or (indice==0) or (" ng-" in element) or ("nginx" in element) or ("ngrx" in element) or ("ngx" in element) )and (element not in list_tag_sup):
        traduction_tags_plus[element].append("ng/ng2/nginx/ngrx/ngx")
 
    if ("pdf" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("pdf")

    try:
        indice=element.index("pip ")
    except ValueError:
        indice=-1
    try:
        indice1=element.index("pip-")
    except ValueError:
        indice1=-1
    try:
        indice2=element.index("pip.")
    except ValueError:
        indice2=-1
    if (" pip " in element or " pip-" in element or " pip." in element or (indice==0) or (indice1==0) or (indice2==0))  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("pip")

    try:
        indice=element.index("py")
    except ValueError:
        indice=-1
    if (indice==0) and  (("python 3." not in element) or ("python3." not in element) or ("python-3." not in element) ) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("python")

    if (("python 3." in element) or ("python3." in element) or ("python-3." in element) ) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("python-3.x")

    try:
        indice=element.index("r ")
    except ValueError:
        indice=-1
    try:
        indice1=element.index("r.")
    except ValueError:
        indice1=-1
    try:
        indice2=element.index("r-")
    except ValueError:
        indice2=-1
    if ((" r " in element) or (indice==0) or (" r." in element) or (indice1==0) or (" r-" in element) or (indice2==0))  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("r")

    if ("rails" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("rails")

    if ("react-" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("reactjs")

    if ("redux" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("redux")

    if ("ruby" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("ruby")
    
    if ("scikit" in element or "scipy" in element or "sklearn" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("python")

    try:
        indice=element.index("sql")
    except ValueError:
        indice=-1
    if ((indice==0) or  (("mysql" not in element)  and ("sql" in element) ) ) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("sql")
        
    if ("symfony" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("symphony")        

    if ("ubuntu" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("ubuntu")

    if ("visual-studio" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("visual-studio")          

    try:
        indice=element.index("vue")
    except ValueError:
        indice=-1
    if ((indice==0) or (" vue " in element)) and (element not in list_tag_sup):
        traduction_tags_plus[element].append("vue.js")

    try:
        indice=element.index("windows")
    except ValueError:
        indice=-1
    if ((indice==0) and ("windows-server" not in element) and (element not in list_tag_sup)):
        traduction_tags_plus[element].append("windows")

    if ("windows-server" in element)  and (element not in list_tag_sup):
        traduction_tags_plus[element].append("visual-studio")
    
    #on teste si les elements de la list_tag_sup se retrouve dans l'element de la liste de tags complete
    for element_sup in list_tag_sup:
        if element_sup in element:
            traduction_tags_plus[element].append(element_sup)

#on "durcit" le dictionnaire
traduction_tags_plus=dict(traduction_tags_plus)

#on copie dans data_dict2 les traductions des tags issus du dictionnaire de traduction
data_dict2=data_dict.copy()
for question in data_dict2:
    for index, value in enumerate(data_dict2[question]['Tags']):
            #on ajoute la traduction du tags
            data_dict2[question]['Tags']=data_dict2[question]['Tags']+traduction_tags_plus[value]
            #on supprime le tag initial
            data_dict2[question]['Tags'].remove(value)
            #on elimine les doublons de la liste
            data_dict2[question]['Tags']=list_unique(data_dict2[question]['Tags'])

#on compte les occurences de la nouvelle base de tags
liste_tags2=[]
for k in data_dict2:
    liste_tags2=liste_tags2+data_dict2[k]['Tags']
from collections import Counter
tag_counter2=Counter(liste_tags2).most_common()

#on supprime des tags en reduisant l'arborescence si possible ex: airflow-->apache          
for question in data_dict2:
    data_dict2[question]['Tags']=list_unique(data_dict2[question]['Tags'])
    for index, value in enumerate(data_dict2[question]['Tags']):
        if value=="airflow":
            data_dict2[question]['Tags'].append("apache")
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="flask":
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="go":
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="maven" :
            data_dict2[question]['Tags'].append("apache")
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="microsoft" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="ng/ng2/nginx/ngrx/ngx" :
            data_dict2[question]['Tags'].append("angular")
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="pdf" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="r" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="rails" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
            data_dict2[question]['Tags'].append("ruby")
        if value=="redux" :
            data_dict2[question]['Tags'].append("javascript")
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value=="symphony" :
            data_dict2[question]['Tags'].append("php")
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
    data_dict2[question]['Tags']=list_unique(data_dict2[question]['Tags'])   

#on compte les occurences de la nouvelle base de tags   
liste_tags3=[]
for k in data_dict2:
    liste_tags3=liste_tags3+data_dict2[k]['Tags']
from collections import Counter
tag_counter3=Counter(liste_tags3).most_common()

#on supprime les tags restants
for question in data_dict2:
    for index, value in enumerate(data_dict2[question]['Tags']):
        if value=="ruby" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))
        if value==".net" :
            data_dict2[question]['Tags']=list(filter((value).__ne__,data_dict2[question]['Tags']))

#on compte les occurences de la nouvelle base de tags 
liste_tags4=[]
for k in data_dict2:
    liste_tags4=liste_tags4+data_dict2[k]['Tags']
from collections import Counter
tag_counter4=Counter(liste_tags4).most_common()


#On supprime de la base les questions qui n'ont pas de tags
liste_question=[]
for k in data_dict2:
    liste_question.append(k)
for question in liste_question:
    if data_dict2[question]['Tags']==[]:
        del data_dict2[question]






###########################################################
#                GESTION DU BODY                          #
###########################################################

#on utilise beautifulsoup


for k in data_dict2:
    #on passe le corps du texte en minuscule et on le copie dans Body2
    data_dict2[k]['Body2']=data_dict2[k]['Body'].lower()
    #on remplace les balises <a> en balises <p>
    value=re.sub('\<a','<p',data_dict2[k]['Body2'])
    data_dict2[k]['Body2']=value
    data_dict2[k]['Body2']=re.sub('\<\/a','</p',data_dict2[k]['Body2'])
    soup = BeautifulSoup(data_dict2[k]['Body2'])
    data_dict2[k]['Body2']=[]
    
    #on remplace les balises pre en p tout en gardant la balise <code>
    for p in soup.find_all('pre'):
        n = BeautifulSoup('<p><code>%s</code></p>' % p.string)
        p.replace_with(n.body.contents[0])    
    #idem pour la balise h1
    for p in soup.find_all('h1'):
        n = BeautifulSoup('<p><h1>%s</h1></p>' % p.string)
        p.replace_with(n.body.contents[0]) 
    #on recupere la liste comprises entre des balises p dans Body2
    for p in soup.find_all('p') :
        data_dict2[k]['Body2'].append(str(p))

#on copie body2 dans body3    
for k in data_dict2:
    data_dict2[k]['Body3']=data_dict2[k]['Body2'].copy()
    
#on boucle sur les questions
for k in data_dict2:    
    #on boucle sur la liste de p extraite
    for index, value in enumerate(data_dict2[k]['Body3']):        
        #on regarde si on trouve la balise code à l'interieur d'un <p>
        #on la modifie de maniere a fermer le p et a en ouvrir un nouveau
        value = value.replace('<code>', '</p><p><code>') 
        value = value.replace('</code>', '</code></p><p>')
        #on soup
        soup2 = BeautifulSoup(value)
        list_temp=[]    
        #on recupere la liste comprises dans les balises p et on l'insere dans la liste précédente
        for q in soup2.find_all('p') :
            list_temp.append(str(q))
        #on supprime les <p> vide 
        list_temp=list(filter(("<p></p>").__ne__,list_temp))
        data_dict2[k]['Body3'][index]=list_temp

#on supprime les elements vide de la liste
for k in data_dict2:    
    for index, value in enumerate(data_dict2[k]['Body3']):        
        if len(value)==0:
            del data_dict2[k]['Body3'][index]
    for index, value in enumerate(data_dict2[k]['Body3']):        
        if len(value)==0:
            del data_dict2[k]['Body3'][index]


#on boucle sur les questions
for k in data_dict2:
    
    data_dict2[k]['Body_texte']=""
    data_dict2[k]['Body_texte_code']=""
    for index, value in enumerate(data_dict2[k]['Body3']):
        for index2, value2 in enumerate(data_dict2[k]['Body3'][index]):
            texte_temp=data_dict2[k]['Body3'][index][index2]
            #on teste si on est sur un element de code ou de lien
            try:
                indice_code=texte_temp.index("<code>")
            except ValueError:
                indice_code=-1
            try:
                indice_href=texte_temp.index("<p href=")
            except ValueError:
                indice_href=-1
                
            #si on est un element du corps, on sauvegarde dans body_texte et body_texte_code
            if indice_code==-1 and indice_href==-1:
                # on gère deux exceptions du texte avant la tokenization
                #"\'" par "'"
                texte_temp = texte_temp.replace('()', '() ')
                texte_temp = texte_temp.replace('''\\\'''','''\'''')
                #on enleve les balises et les sauts de lignes
                texte_temp = texte_temp.replace('\n', ' ')
                texte_temp = texte_temp.replace('<strong>', '')
                texte_temp = texte_temp.replace('</strong>', '')
                texte_temp = texte_temp.replace('<br>', '')
                texte_temp = texte_temp.replace('</br>', '')
                texte_temp = texte_temp.replace('<br/>', '')
                texte_temp = texte_temp.replace('</p>', '')
                texte_temp = texte_temp.replace('<p>', '')
                data_dict2[k]['Body_texte']=data_dict2[k]['Body_texte']+texte_temp
                data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+texte_temp
            #si on est un element du code, on sauvegarde dans body_texte_code            
            if indice_code!=-1 and indice_href==-1:
                             
                #on enleve les balises et les sauts de lignes
                texte_temp = texte_temp.replace('\n', ' ')
                texte_temp = texte_temp.replace('<strong>', '')
                texte_temp = texte_temp.replace('</strong>', '')
                texte_temp = texte_temp.replace('<br>', '')
                texte_temp = texte_temp.replace('</br>', '')
                texte_temp = texte_temp.replace('<br/>', '')
                texte_temp = texte_temp.replace('</p>', '')
                texte_temp = texte_temp.replace('<p>', '')
                texte_temp = texte_temp.replace('</code>', '')
                texte_temp = texte_temp.replace('<code>', '')            
                data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+texte_temp    

#une première tokenization rapide sans ponctuation
tokenizer = nltk.RegexpTokenizer(r'\w+')
corpora = defaultdict(list)
# Création d'un corpus de tokens par question
for question in data_dict2:
        corpora[question] = tokenizer.tokenize(
                                data_dict2[question]['Body_texte'].lower()
                            )

#nombre de mots dans texte par question
stats, freq = dict(), dict()
for k, v in corpora.items():
    freq[k] = fq = nltk.FreqDist(v)
    stats[k] = {'total': len(v)}
# Récupération des comptages
nombre_mot_body_texte_par_question = pd.DataFrame.from_dict(stats, orient='index')
# Affichage des fréquences
nombre_mot_body_texte_par_question=nombre_mot_body_texte_par_question.sort_values(by='total', ascending=False)
nombre_mot_body_texte_par_question[0:49].plot(kind='bar', color="#f56900", title='Top 50 du nombre de mot par question')
data2=pd.DataFrame.from_dict(data_dict, orient='index')


#nombre de mots uniques dans texte par question
stats_unique, freq_unique = dict(), dict()

for k, v in corpora.items():
    freq_unique[k] = fq = nltk.FreqDist(v)
    stats_unique[k] = {'total': len(v), 'unique': len(fq.keys())}

df_unique = pd.DataFrame.from_dict(stats_unique, orient='index')

# Affichage des fréquences
df_unique=df_unique.sort_values(by='total', ascending=False)
df_unique[0:49].plot(kind='bar', color="#f56900", title='Top 50 du nombre de mot unique par question')

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
most_freq = list(zip(*freq_totale.most_common(16)))[0]

nltk.download('stopwords')
# On créé notre set de stopwords final qui cumule ainsi les 100 mots les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut présent dans la librairie NLTK
sw = set()
sw.update(most_freq)
sw.update(tuple(nltk.corpus.stopwords.words('english')))
tuple_sw_ajout=('I','ca','''can't''',''''s''')
sw.update(tuple_sw_ajout)

#ajout du lemmatizer
import spacy
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")
print(lemmatizer.mode)

#ajout de la racinisation nlp 

suffixes = list(nlp.Defaults.suffixes)
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

prefixes = list(nlp.Defaults.prefixes)
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search
 

#on tokenize
from autocorrect import Speller
import time
import unicodedata
t1=time.time()

#Enlever le Fast=True

#spell = Speller()
body_sans_stopwords_lemme_stem_sans_nltk_speller = defaultdict(list)
for question in data_dict2:
    #on enleve les accents
    data_dict2[question]['Body_texte']=str(unicodedata.normalize('NFD', data_dict2[question]['Body_texte']).encode('ascii', 'ignore'))[2:-1] 
    #data_dict2[question]['Body_texte']=spell(data_dict2[question]['Body_texte'].lower())
    #on passe en minuscule
    data_dict2[question]['Body_texte']=data_dict2[question]['Body_texte'].lower()
    doc=nlp(data_dict2[question]['Body_texte'])
    tokens=[token.lemma_ for token in doc]
    #on enleve les mots compris dans sw
    body_sans_stopwords_lemme_stem_sans_nltk_speller[question] += [w for w in tokens if (not w in list(sw)) ]

#on sauvegarde
import pickle
with open("myDictionary.pkl", "wb") as tf:
    pickle.dump(body_sans_stopwords_lemme_stem_sans_nltk_speller,tf)


with open("myDictionary.pkl", "rb") as tf:
    body_sans_stopwords_lemme_stem_sans_nltk_speller = pickle.load(tf)

    
for question in body_sans_stopwords_lemme_stem_sans_nltk_speller:
    index=0
    while index<len(body_sans_stopwords_lemme_stem_sans_nltk_speller[question]):
        #drapeau pour dire qu'on a effectué une opération et donc que l'on incrémente pas l'index pour retester
        drap=0
        value=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
        #on supprime si la valeur est vide
        if value=="":
            del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        
        #on regarde si le mot suivant n'est pas "(" et celui d'après ")"
        #si c'est le cas, on agrege "()" au premier mot et on supprime les deux suivants
        if (index+1)<=(len(body_sans_stopwords_lemme_stem_sans_nltk_speller[question])-1) and drap==0:
            if body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]==")" and value=="(":
                body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]+"()"
                del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]
                del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                drap=1
        #on supprime "n't"        
        if value=="n't" and drap==0:
            del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        
        #on remplace les \' par '
        try:
            indice=value.index('''\\\'''')
        except ValueError:
            indice=-1
        if indice!=-1 and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].replace('''\\\'''','''\'''')
            drap=1

        #on remplace pe par ping
        if value=="pe" and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="ping"
            drap=1
        
        #on supprime le . s'il est en début ou en fin de mot
        try:
            indice=value.index(".")
        except ValueError:
            indice=-1
        if (indice==0 or indice==len(value)-1) and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index] = value.replace('.', '')
            drap=1
        
        #on supprime le ? s'il est en début ou en fin de mot
        try:
            indice=value.index("?")
        except ValueError:
            indice=-1
        if (indice==0 or indice==len(value)-1) and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index] = value.replace('?', '')
            drap=1
        
        #on teste si l'expression ne contient pas les mots suivants mais contient un . et pas de ()
        #on teste aussi que cela soit pas une adresse ip ou un numero de version
        try:
            indice=value.index(".com")
        except ValueError:
            indice=-1
        try:
            indice1=value.index(".org")
        except ValueError:
            indice1=-1
        try:
            indice2=value.index(".fr")
        except ValueError:
            indice2=-1        
        try:
            indice2=value.index(".net")
        except ValueError:
            indice2=-1        
        try:
            indice3=value.index(".exe")
        except ValueError:
            indice3=-1        
        try:
            indice4=value.index(".ini")
        except ValueError:
            indice4=-1        
        try:
            indice5=value.index(".bat")
        except ValueError:
            indice5=-1        
        try:
            indice6=value.index(".py")
        except ValueError:
            indice6=-1        
        try:
            indice7=value.index(".js")
        except ValueError:
            indice7=-1        
        try:
            indice8=value.index("()")
        except ValueError:
            indice8=-1         
        
        try:
            indice9=value.index(".")
        except ValueError:
            indice9=-1        
        if indice==-1 and indice1==-1 and indice2==-1 and indice3==-1 and indice4==-1 and indice5==-1 and indice6==-1 and indice7==-1 and indice8==-1 and indice9!=-1 and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.x",value) is None and re.match(r"\d{1,3}\.X",value) is None and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and drap==0:    
            #il faut donc séparer les mots par le .
            list_temp=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].split(".")
            #on crée une nouvelle liste que l'on insere dans la liste mere
            for index2, value2 in enumerate(list_temp):
                if index2==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=list_temp[index2]
                else:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question].insert(index+index2, list_temp[index2])
            drap=1
        
        #on teste si l'expression contient un ?
        try:
            indice=value.index("?")
        except ValueError:
            indice=-1        
        if  indice!=-1 and drap==0:    
            #il faut donc séparer les mots par le ?
            list_temp=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].split("?")
            #on crée une nouvelle liste que l'on insere dans la liste mere
            for index2, value2 in enumerate(list_temp):
                if index2==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=list_temp[index2]
                else:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question].insert(index+index2, list_temp[index2])
            drap=1
   
        #on teste si le mot est composé d'espace
        y=""
        x=0
        while x<20:
            if value==y and drap==0:
                del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                drap=1
            y=y+" "
            x=x+1
        #on teste si le mot est de longueur 1 et est non alphanumerique
        if (re.search("[a-zA-Z0-9]",value) is None or value=="," or value=="i") and len(value)==1 and drap==0:
            del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        
        #si on a effectué aucune modification, on peut augmenter l'index
        if drap==0:
            index=index+1           
            
t2=time.time()
print ("la tokenization des body prend {} min à tourner".format((t2-t1)/60))

#statistiques sur les body
stats_body_sans_stopwords_lemme_stem_sans_nltk_speller, freq_body_sans_stopwords_lemme_stem_sans_nltk_speller = dict(), dict()

for k, v in body_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    freq_body_sans_stopwords_lemme_stem_sans_nltk_speller[k] = fq = nltk.FreqDist(v)
    stats_body_sans_stopwords_lemme_stem_sans_nltk_speller[k] = {'total': len(v), 'unique': len(fq.keys())}

df_body_unique_sans_stopwords_lemme_stem_sans_nltk_speller = pd.DataFrame.from_dict(stats_body_sans_stopwords_lemme_stem_sans_nltk_speller, orient='index')

# Affichage des fréquences
df_body_unique_sans_stopwords_lemme_stem_sans_nltk_speller=df_body_unique_sans_stopwords_lemme_stem_sans_nltk_speller.sort_values(by='total', ascending=False)
df_body_unique_sans_stopwords_lemme_stem_sans_nltk_speller[0:49].plot(kind='bar', color="#f56900", title='Top 50 des body par nombre de mots uniques')

#on cree une nouvelle clé dans data_dict2 et on fait une liste des body pour les modélisations ultérieures
list_body_model=[]
for question, value in body_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    body=" ".join(value)
    data_dict2[question]['Body_token']=body
    list_body_model.append(body)




###########################################################
#                GESTION DE LA QUESTION                   #
###########################################################



#on tokenize
#on garde les prefixe suffixes précédents
#on garde aussi les stopword

t1=time.time()
question_sans_stopwords_lemme_stem_sans_nltk_speller = defaultdict(list)
for question in data_dict2:
    #on enleve les accents
    data_dict2[question]['question']=str(unicodedata.normalize('NFD', question).encode('ascii', 'ignore'))[2:-1] 
    #data_dict2[question]['Body_texte']=spell(data_dict2[question]['Body_texte'].lower())
    #on passe en minuscule
    data_dict2[question]['question']=data_dict2[question]['question'].lower()
    doc=nlp(data_dict2[question]['question'])
    tokens=[token.lemma_ for token in doc]
    question_sans_stopwords_lemme_stem_sans_nltk_speller[question] += [w for w in tokens if (not w in list(sw)) ]

#on sauvegarde
import pickle
with open("myDictionary2.pkl", "wb") as tf:
    pickle.dump(question_sans_stopwords_lemme_stem_sans_nltk_speller,tf)

  
with open("myDictionary2.pkl", "rb") as tf:
    question_sans_stopwords_lemme_stem_sans_nltk_speller = pickle.load(tf)    

for question in question_sans_stopwords_lemme_stem_sans_nltk_speller:
    index=0
    while index<len(question_sans_stopwords_lemme_stem_sans_nltk_speller[question]):
        drap=0
        value=question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
        #on remplace \' par ' 
        try:
            indice=value.index('''\\\'''')
        except ValueError:
            indice=-1
        if indice!=-1:
            question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].replace('''\\\'''','''\'''')
            drap=1
        #on regarde si le mot suivant n'est pas "(" et celui d'après ")"
        #si c'est le cas, on agrege "()" au premier mot et on supprime les deux suivants
        if (index+1)<=(len(question_sans_stopwords_lemme_stem_sans_nltk_speller[question])-1) and drap==0:
            if question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]==")" and value=="(":
                question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]=question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]+"()"
                del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]
                del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                drap=1
                
        #on regarde si le mot n'est pas de taille 1 à 5 avec que du non alphanumerique
        if (re.search("[a-zA-Z0-9]",value) is None or value=="i") and len(value)==1 and drap==0:
            del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        if (re.search("[a-zA-Z0-9]{2}",value) is None) and len(value)==2 and drap==0:
            del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        if (re.search("[a-zA-Z0-9]{3}",value) is None) and len(value)==3 and drap==0:
            del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        if (re.search("[a-zA-Z0-9]{4}",value) is None) and len(value)==4 and drap==0:
            del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        if (re.search("[a-zA-Z0-9]{5}",value) is None) and len(value)==5 and drap==0:
            del question_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
            drap=1
        
        #si pas de modification, on incrémente l'index
        if drap==0:
            index=index+1
                
t2=time.time()
print ("la tokenization des questions prend {} min à tourner".format((t2-t1)/60))

#on effectue des statistiques
stats_question_sans_stopwords_lemme_stem_sans_nltk_speller, freq_question_sans_stopwords_lemme_stem_sans_nltk_speller = dict(), dict()

for k, v in question_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    freq_question_sans_stopwords_lemme_stem_sans_nltk_speller[k] = fq = nltk.FreqDist(v)
    stats_question_sans_stopwords_lemme_stem_sans_nltk_speller[k] = {'total': len(v), 'unique': len(fq.keys())}

df_question_unique_sans_stopwords_lemme_stem_sans_nltk_speller = pd.DataFrame.from_dict(stats_question_sans_stopwords_lemme_stem_sans_nltk_speller, orient='index')

# Affichage des fréquences
df_question_unique_sans_stopwords_lemme_stem_sans_nltk_speller=df_question_unique_sans_stopwords_lemme_stem_sans_nltk_speller.sort_values(by='total', ascending=False)
#df_question_unique_sans_stopwords_lemme_stem_sans_nltk_speller[0:49].plot(kind='bar', color="#f56900", title='Top 50 des body par nombre de mots uniques')

#on cree une nouvelle clé dans data_dict2 et on fait une liste des questions pour les modélisations ultérieures
list_question_model=[]
for question, value in question_sans_stopwords_lemme_stem_sans_nltk_speller.items():
    question2=" ".join(value)
    data_dict2[question]['question_token']=question2
    list_question_model.append(question2)

###########################################################
#                    STATISTIQUES                         #
###########################################################

#calcul du nombre de mot par question et par body_texte brut et après tokenization
df_total= pd.DataFrame.from_dict(data_dict2, orient='index')
df_total=df_total[["Tags","question_token","Body_token","Body_texte"]]
df_total['question']=df_total.index
df_total['len_question'] = [len(x.split()) for x in df_total['question'].tolist()]
df_total['len_question_token'] = [len(x.split()) for x in df_total['question_token'].tolist()]
df_total['len_Body_texte'] = [len(x.split()) for x in df_total['Body_texte'].tolist()]
df_total['len_Body_token'] = [len(x.split()) for x in df_total['Body_token'].tolist()]

import matplotlib.pyplot as plt

plt.hist(df_total['len_question'],bins=10)
plt.title('distribution du nombre de mots dans la question initiale')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.hist(df_total['len_question_token'],bins=10)
plt.title('distribution du nombre de mots dans la question tokenize')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.hist(df_total[df_total['len_Body_texte']<200]['len_Body_texte'],bins=10)
plt.title('distribution du nombre de mots dans le body')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.hist(df_total[df_total['len_Body_token']<200]['len_Body_token'],bins=10)
plt.title('distribution du nombre de mots dans le body tokenize')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()




###########################################################
#                    MODELISATION                         #
###########################################################


#LDA body

from sklearn.feature_extraction.text import CountVectorizer
#max_features=nombre de mots que l'on va garder dans le vocabulaire
tf_vectorizer=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
tf_body=tf_vectorizer.fit_transform(list_body_model)

    
from sklearn.decomposition import LatentDirichletAllocation
#a tester hyperparametres nbr de sujets
n_topics = np.arange(10,101,5)
data_topic_body_lda = pd.DataFrame(index=np.arange(100))
for n_topic in n_topics:
    print(n_topic)
    # Créer le modèle LDA
    lda = LatentDirichletAllocation(
            n_components=n_topic, 
            max_iter=5, 
            learning_method='online', 
            learning_offset=50.,
            random_state=0)
    
    # Fitter sur les données
    lda.fit(tf_body)
    no_top_words=20
    feature_names=tf_vectorizer.get_feature_names()
    data_topic_body_lda['topic_feature_lda_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(lda.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic_body_lda['topic_feature_lda_'+str(n_topic)][topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])


#NMF Body

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(list_body_model)


n_topics = np.arange(10,101,5)
data_topic_body_nmf = pd.DataFrame(index=np.arange(100))
for n_topic in n_topics:

    # Run NMF
    nmf = NMF(n_components=n_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
    nmf.fit(tfidf)

    no_top_words=20
    feature_names = tfidf_vectorizer.get_feature_names()
    data_topic_body_nmf['topic_feature_nmf_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(nmf.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic_body_nmf['topic_feature_nmf_'+str(n_topic)][topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])


#LDA question

#max_features=nombre de mots que l'on va garder dans le vocabulaire
tf_vectorizer=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
tf_question=tf_vectorizer.fit_transform(list_question_model)

#a tester hyperparametres nbr de sujets
n_topics = np.arange(10,101,5)
data_topic_question_lda = pd.DataFrame(index=np.arange(100))
for n_topic in n_topics:
    # Créer le modèle LDA
    lda = LatentDirichletAllocation(
            n_components=n_topic, 
            max_iter=5, 
            learning_method='online', 
            learning_offset=50.,
            random_state=0)
    
    # Fitter sur les données
    lda.fit(tf_question)
    no_top_words=20
    feature_names=tf_vectorizer.get_feature_names()  
    data_topic_question_lda['topic_feature_lda_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(lda.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic_question_lda['topic_feature_lda_'+str(n_topic)][topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        

#NMF question    

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(list_question_model)


n_topics = np.arange(10,101,5)
data_topic_question_nmf = pd.DataFrame(index=np.arange(100))
for n_topic in n_topics:

    # Run NMF
    nmf = NMF(n_components=n_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd',max_iter=1000)
    nmf.fit(tfidf)

    no_top_words=20
    feature_names = tfidf_vectorizer.get_feature_names()
    data_topic_question_nmf['topic_feature_nmf_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(nmf.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic_question_nmf['topic_feature_nmf_'+str(n_topic)][topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        
    