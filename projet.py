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
    Cette fonction prend une liste en entrée et retourne la liste sans doublon
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
    #data_dict2[k]['Body2']=re.sub('\<\/a','</p',data_dict2[k]['Body'])

    #on passe le corps du texte en minuscule et on le copie dans Body2
    data_dict2[k]['Body2']=data_dict2[k]['Body'].lower()
    #on remplace les balises <a> en balises <p>
    #value=re.sub('\<a','<p',data_dict2[k]['Body2'])
    #data_dict2[k]['Body2']=value
    #data_dict2[k]['Body2']=re.sub('\<\/a','</p',data_dict2[k]['Body2'])
    soup = BeautifulSoup(data_dict2[k]['Body2'])
    data_dict2[k]['Body2']=[]
    
    #on remplace les balises pre en p tout en gardant la balise <code>
    for p in soup.find_all('pre'):
        n = BeautifulSoup('<p><code>%s</code></p>' % p.string)
        p.replace_with(n.body.contents[0])    
    
    #idem pour la balise h1
    for p in soup.find_all('h1'):
        n = BeautifulSoup('<p>%s</p>' % p.string)
        p.replace_with(n.body.contents[0]) 
    for p in soup.find_all('h2'):
        n = BeautifulSoup('<p>%s</p>' % p.string)
        p.replace_with(n.body.contents[0])     #on recupere la liste comprises entre des balises p dans Body2
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
        value = value.replace('<h2>', '</p><p><h2>') 
        value = value.replace('</h2>', '</h2></p><p>')
        try:
            indice_a=value.index("<a")
        except ValueError:
            indice_a=-1       
        if indice_a!=3 and indice_a!=-1:
            value = value.replace('<a', '</p><p><a') 
            value = value.replace('</a>', '</a></p><p>')
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
                indice_href=texte_temp.index("<a href=")
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
                texte_temp = texte_temp.replace('<h1>', '')
                texte_temp = texte_temp.replace('</h1>', '')
                texte_temp = texte_temp.replace('<h2>', '')
                texte_temp = texte_temp.replace('</h2>', '')
                data_dict2[k]['Body_texte']=data_dict2[k]['Body_texte']+" "+texte_temp
                data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+" "+texte_temp
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
                texte_temp = texte_temp.replace('<h1>', '')
                texte_temp = texte_temp.replace('</h1>', '')
                texte_temp = texte_temp.replace('<h2>', '')
                texte_temp = texte_temp.replace('</h2>', '')
                texte_temp = texte_temp.replace('</code>', '')
                texte_temp = texte_temp.replace('<code>', '')            
                data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+" "+texte_temp    
    data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code'][1:]
    data_dict2[k]['Body_texte']=data_dict2[k]['Body_texte'][1:]
    
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

nltk.download('stopwords')
# On créé notre set de stopwords final qui cumule ainsi les 100 mots les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut présent dans la librairie NLTK
sw = set()
sw.update(most_freq)
sw.update(tuple(nltk.corpus.stopwords.words('english')))
tuple_sw_ajout=('I','ca','''can't''',''''s''','know','run','want','use','try','error','like','question','issue','code','example','solution')
sw.update(tuple_sw_ajout)
tuple_sw_discard=('c','project','app','file','1','android','0', 'version','2','data', 'js','new','3','3', 'java','component', 'below', 'class', 'artifactid', 'getting', 'function','flutter','build')
for dis in tuple_sw_discard:
    sw.discard(dis)
print(sw)

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


data_fname = 'QueryResults.csv'
data = pd.read_csv(os.path.join(path_to_data, data_fname), sep=",")
#on sauvegarde
import pickle
data_fname = 'myDictionary.pkl'
with open(os.path.join(path_to_data, data_fname), "wb") as tf:
    pickle.dump(body_sans_stopwords_lemme_stem_sans_nltk_speller,tf)


data_fname = 'myDictionary.pkl'
with open(os.path.join(path_to_data, data_fname), "rb") as tf:
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
        #on corrige instal
        if value=="instal" and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="install"
            drap=1
        #on corrige jupiter
        if value=="jupiter" and drap==0:
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="jupyter"
            drap=1
        #on supprime le . s'il est en début ou en fin de mot
        try:
            indice=value.index(".")
        except ValueError:
            indice=-1
        if (indice==0 or indice==len(value)-1) and value!=".com" and value!=".org" and value!=".fr" and value!=".net" and value!=".exe" and value!=".ini" and value!=".bat" and value!=".py" and value!=".js" and value!=".net-core" and drap==0:
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
            indice8=value.index(".net-core")
        except ValueError:
            indice8=-1        
        try:
            indice9=value.index("()")
        except ValueError:
            indice9=-1         
        
        try:
            indice10=value.index(".")
        except ValueError:
            indice10=-1        
        if indice==-1 and indice1==-1 and indice2==-1 and indice3==-1 and indice4==-1 and indice5==-1 and indice6==-1 and indice7==-1 and indice8==-1 and indice9==-1 and indice10!=-1 and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.x",value) is None and re.match(r"\d{1,3}\.X",value) is None and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and drap==0:    
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
data_fname = 'myDictionary2.pkl'
with open(os.path.join(path_to_data, data_fname), "wb") as tf:
    pickle.dump(question_sans_stopwords_lemme_stem_sans_nltk_speller,tf)


data_fname = 'myDictionary2.pkl'
with open(os.path.join(path_to_data, data_fname), "rb") as tf:
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
df_total['nbr_tag'] = [len(x) for x in df_total['Tags']]

effectifs = df_total['nbr_tag'].value_counts()
modalites = effectifs.index # l'index de effectifs contient les modalités

df_nbr_tag = pd.DataFrame(modalites, columns = ['nbr_tag']) # création du tableau à partir des modalités
df_nbr_tag["n"] = effectifs.values
df_nbr_tag["f"] = df_nbr_tag["n"] / len(df_total) # len(data) renvoie la taille de l'échantillon

import matplotlib.pyplot as plt
plt.figure()
plt.hist(df_total['len_question'],bins=10)
plt.title('distribution du nombre de mots dans la question initiale')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.figure()
plt.hist(df_total['len_question_token'],bins=10)
plt.title('distribution du nombre de mots dans la question tokenize')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.figure()
plt.hist(df_total[df_total['len_Body_texte']<200]['len_Body_texte'],bins=10)
plt.title('distribution du nombre de mots dans le body')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()

plt.figure()
plt.hist(df_total[df_total['len_Body_token']<200]['len_Body_token'],bins=10)
plt.title('distribution du nombre de mots dans le body tokenize')
plt.xlabel('nombre de mots')
plt.ylabel('nombre de questions')
plt.show()


list_question_brute=[]
for question in data_dict2:
    list_question_brute.append(question)

###########################################################
#                    MODELISATION                         #
###########################################################



#LDA body

from sklearn.feature_extraction.text import CountVectorizer
#max_features=nombre de mots que l'on va garder dans le vocabulaire
tf_vectorizer_body=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
tf_body=tf_vectorizer_body.fit_transform(list_body_model)

#Sparsicity is nothing but the percentage of non-zero datapoints in the document-word matrix, that is data_vectorized.
#Since most cells in this matrix will be zero, I am interested in knowing what percentage of cells 
#contain non-zero values.

# Materialize the sparse data
tf_body_dense = tf_body.todense()
# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((tf_body_dense > 0).sum()/tf_body_dense.size)*100, "%")
    
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
t1=time.time()
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
#search_params = {'n_components': [10], 'learning_decay': [.7]}
lda = LatentDirichletAllocation(max_iter=5, 
             learning_method='online', 
             learning_offset=50.,
             random_state=0)
model = GridSearchCV(lda, param_grid=search_params,cv=5)
model.fit(tf_body)
t2=time.time()
print ("la lda body prend {} min à tourner".format((t2-t1)/60))
lda_body_best_model = model.best_estimator_
print("Best Model's Params lda tf_body: ", model.best_params_)
lda_body_best_params=model.best_params_
print("Best Log Likelihood Score: ", model.best_score_)
no_top_words=20
feature_names=tf_vectorizer_body.get_feature_names()
nbr_topic_lda_retenu=lda_body_best_params['n_components']
data_topic_body_lda= pd.DataFrame(index=np.arange(nbr_topic_lda_retenu))
data_topic_body_lda['topic_feature_lda_'+str(nbr_topic_lda_retenu)]=""
for topic_idx, topic in enumerate(lda_body_best_model.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    data_topic_body_lda['topic_feature_lda_'+str(nbr_topic_lda_retenu)][topic_idx]=[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]




#NMF Body
#NMF can't be scored (at least in scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

# NMF is able to use tf-idf
tfidf_vectorizer_body = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
tfidf_body = tfidf_vectorizer_body.fit_transform(list_body_model)


n_topics = np.arange(10,31,5)
data_topic_body_nmf = pd.DataFrame(index=np.arange(30))
for n_topic in n_topics:

    # Run NMF
    nmf = NMF(n_components=n_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd',max_iter=1000)
    nmf.fit(tfidf_body)

    no_top_words=20
    feature_names = tfidf_vectorizer_body.get_feature_names()
    data_topic_body_nmf['topic_feature_nmf_'+str(n_topic)]=""
    for topic_idx, topic in enumerate(nmf.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        data_topic_body_nmf['topic_feature_nmf_'+str(n_topic)][topic_idx]=[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]


# =============================================================================
# #LDA question
# 
# #max_features=nombre de mots que l'on va garder dans le vocabulaire
# tf_vectorizer_question=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
# tf_question=tf_vectorizer_question.fit_transform(list_question_model)
# 
# # Materialize the sparse data
# tf_question_dense = tf_question.todense()
# # Compute Sparsicity = Percentage of Non-Zero cells
# print("Sparsicity: ", ((tf_question_dense > 0).sum()/tf_question_dense.size)*100, "%")
#     
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.model_selection import GridSearchCV
# t1=time.time()
# search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
# lda = LatentDirichletAllocation(max_iter=5, 
#              learning_method='online', 
#              learning_offset=50.,
#              random_state=0)
# model = GridSearchCV(lda, param_grid=search_params,cv=5)
# model.fit(tf_question)
# t2=time.time()
# print ("la lda question prend {} min à tourner".format((t2-t1)/60))
# lda_question_best_model = model.best_estimator_
# print("Best Model's Params lda tf_question: ", model.best_params_)
# lda_question_best_params=model.best_params_
# feature_names=tf_vectorizer_question.get_feature_names()
# nbr_topic_question_lda_retenu=lda_question_best_params['n_components']
# data_topic_question_lda= pd.DataFrame(index=np.arange(nbr_topic_question_lda_retenu))
# data_topic_question_lda['topic_feature_lda_'+str(nbr_topic_question_lda_retenu)]=""
# for topic_idx, topic in enumerate(lda_question_best_model.components_):
# #        print("Topic {}:".format(topic_idx))
# #        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
#     data_topic_question_lda['topic_feature_lda_'+str(nbr_topic_question_lda_retenu)][topic_idx]=[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
# 
# =============================================================================

# =============================================================================
# #NMF question    
# 
# # NMF is able to use tf-idf
# tfidf_vectorizer_question = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
# tfidf_question = tfidf_vectorizer_question.fit_transform(list_question_model)
# 
# 
# n_topics = np.arange(10,31,5)
# data_topic_question_nmf = pd.DataFrame(index=np.arange(30))
# for n_topic in n_topics:
# 
#     # Run NMF
#     nmf = NMF(n_components=n_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd',max_iter=1000)
#     nmf.fit(tfidf_question)
# 
#     no_top_words=20
#     feature_names = tfidf_vectorizer_question.get_feature_names()
#     data_topic_question_nmf['topic_feature_nmf_'+str(n_topic)]=""
#     for topic_idx, topic in enumerate(nmf.components_):
# #        print("Topic {}:".format(topic_idx))
# #        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
#         data_topic_question_nmf['topic_feature_nmf_'+str(n_topic)][topic_idx]=[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
#         
# =============================================================================



#on sauvegarde

data_fname = 'myDictionary3.pkl'
with open(os.path.join(path_to_data, data_fname), "wb") as tf:
    pickle.dump(data_dict2,tf)


data_fname = 'myDictionary3.pkl'
with open(os.path.join(path_to_data, data_fname), "rb") as tf:
    data_dict2 = pickle.load(tf)  
    

###########################################################
#                ON GARDE LDA A 10 TOPICS                 #
###########################################################

#on calcule le nombre de mot du topic dans la question    

for question in data_dict2:
    index_topic=0
    while index_topic<=nbr_topic_lda_retenu-1:
        a=0
        list_topic=data_topic_body_lda['topic_feature_lda_'+str(nbr_topic_lda_retenu)][index_topic]
        for element in list_topic:
            data_dict2[question]['Body_token2']=data_dict2[question]['Body_token'].split(" ")
            if element in data_dict2[question]['Body_token2']:     
                a=a+1
        data_dict2[question]['topic_'+str(index_topic)]=a
        data_dict2[question]['topic_'+str(index_topic)]=data_dict2[question]['topic_'+str(index_topic)]/20
        index_topic=index_topic+1

###########################################################
#                T SNE A PARTIR DE TF_BODY                #
###########################################################


data3=pd.DataFrame(list_question_brute,columns = ['question'])
from sklearn import manifold
n_components = 2
perplexities = list(range(0, 51, 5))
for j, perplexity in enumerate(perplexities):
    print('j={}'.format(j))
    tsne = manifold.TSNE(n_components=n_components,random_state=1, perplexity=perplexity)
    a=time.time()
    result_temp = tsne.fit_transform(tf_body,list_question_brute)
    b=time.time()
    print('temps pour fitter avec une perplexité de {0}: {1}'.format(perplexity,((b-a)/60)))
    temp=pd.DataFrame(result_temp, columns=["data_tsne_results_"+str(perplexity)+"_F"+str(i+1) for i in range(2)] )
    data3=pd.concat([data3,temp],axis=1)


df_total['Tags_str'] = [' '.join(map(str, l)) for l in df_total['Tags']]
df_total['Tags_str']=df_total['Tags_str']+" "
df_total["tag_java"]=df_total.Tags_str.str.contains("java ")
df_total["tag_python"]=df_total.Tags_str.str.contains("python ")
df_total_tsne_tf=pd.merge(df_total,data3,how='left',on='question')

for j, perplexity in enumerate(perplexities):
    colors = ['c', 'b']
    
    plt.figure()
    non_java = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_java']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_tf[df_total_tsne_tf['tag_java']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    java = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_tf[df_total_tsne_tf['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((java, non_java),
               ('java', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne countvectorized avec une perplexité="+str(perplexity))
    plt.show()
    colors = ['y', 'b']
    
    plt.figure()
    non_python = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_python']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_tf[df_total_tsne_tf['tag_python']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    python = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_tf[df_total_tsne_tf['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((python, non_python),
               ('python', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne countvectorized avec une perplexité="+str(perplexity))
    plt.show()

    colors = ['c', 'y']
    plt.figure()
    java = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_tf[df_total_tsne_tf['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])
    python = plt.scatter(df_total_tsne_tf[df_total_tsne_tf['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_tf[df_total_tsne_tf['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])

    plt.legend((python, java),
               ('python', 'java'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne countvectorized avec une perplexité="+str(perplexity))
    plt.show()   
###########################################################
#                T SNE A PARTIR DE LA MATRICE             #
###########################################################

df_total_matrix= pd.DataFrame.from_dict(data_dict2, orient='index')
liste_topic=[]
x=0
while x<=nbr_topic_lda_retenu-1:
    liste_topic.append("topic_"+str(x))
    x=x+1
data_tsne_matrix=df_total_matrix[liste_topic]
df_total_matrix['question']=df_total_matrix.index

data_topic_body_lda['mean_topic']=0
x=0
while x<nbr_topic_lda_retenu:
    data_topic_body_lda.loc[x, 'mean_topic'] = data_tsne_matrix["topic_"+str(x)].mean()
    x=x+1
temp=data_tsne_matrix.copy()
temp['nbr_topic']=0
x=0
while x<nbr_topic_lda_retenu:
    temp["drap_topic_"+str(x)]=np.where( temp["topic_"+str(x)] > 0.33, 1, 0)
    temp['nbr_topic']=temp['nbr_topic']+temp["drap_topic_"+str(x)]
    x=x+1 

plt.figure()
plt.hist(temp['nbr_topic'],bins=10)
plt.title('distribution du nombre de topic par question')
plt.xlabel('nombre de topic par question')
plt.ylabel('nombre de questions')
plt.show()
    
data3=pd.DataFrame(list_question_brute,columns = ['question'])
from sklearn import manifold
n_components = 2
perplexities = list(range(0, 51, 5))
for j, perplexity in enumerate(perplexities):
    print('j={}'.format(j))
    tsne = manifold.TSNE(n_components=n_components,random_state=1, perplexity=perplexity)
    a=time.time()
    result_temp = tsne.fit_transform(data_tsne_matrix,list_question_brute)
    b=time.time()
    print('temps pour fitter avec une perplexité de {0}: {1}'.format(perplexity,((b-a)/60)))
    temp=pd.DataFrame(result_temp, columns=["data_tsne_results_"+str(perplexity)+"_F"+str(i+1) for i in range(2)] )
    data3=pd.concat([data3,temp],axis=1)

df_total_matrix['Tags_str'] = [' '.join(map(str, l)) for l in df_total_matrix['Tags']]
df_total_matrix['Tags_str']=df_total_matrix['Tags_str']+" "
df_total_matrix["tag_java"]=df_total_matrix.Tags_str.str.contains("java ")
df_total_matrix["tag_python"]=df_total_matrix.Tags_str.str.contains("python ")


df_total_tsne_matrix=pd.merge(df_total_matrix,data3,how='left',on='question')


for j, perplexity in enumerate(perplexities):
    colors = ['c', 'b']
    
    plt.figure()
    non_java = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    java = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((java, non_java),
               ('java', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne (matrice 10) avec une perplexité="+str(perplexity))
    plt.show()
    colors = ['y', 'b']
    
    plt.figure()
    non_python = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==False]\
                                    ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    python = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((python, non_python),
               ('python', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne (matrice 10) avec une perplexité="+str(perplexity))
    plt.show()
    colors = ['c', 'y']
    plt.figure()
    java = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_matrix[df_total_tsne_matrix['tag_java']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])
    python = plt.scatter(df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_matrix[df_total_tsne_matrix['tag_python']==True]\
                        ['data_tsne_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])

    plt.legend((python, java),
               ('python', 'java'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne (matrice 10) avec une perplexité="+str(perplexity))
    plt.show()

###########################################################
#                ACP A PARTIR DE LA MATRICE               #
###########################################################
from sklearn import decomposition
from sklearn import preprocessing
from matplotlib.collections import LineCollection

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


# choix du nombre de composantes à calculer
n_comp = nbr_topic_lda_retenu

# selection des colonnes à prendre en compte dans l'ACP
data_pca=data_tsne_matrix.copy()

# préparation des données pour l'ACP
#data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
X = data_pca.values
names = data_pca.index # ou data.index pour avoir les intitulés
features = data_pca.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

# Eboulis des valeurs propres
display_scree_plot(pca)

# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))
#,(2,3),(4,5)
# Projection des individus
X_pca_projected = pca.transform(X_scaled)
X_pca_project=pd.DataFrame(X_pca_projected, index=names,columns=["F"+str(i+1) for i in range(n_comp)] )
explained_variance_sum=0
for i in range(n_comp):
    explained_variance_sum=explained_variance_sum+100*pca.explained_variance_ratio_[i]
    if explained_variance_sum>90:
        del X_pca_project['F'+str(i+1)]
liste_axes_retenus=X_pca_project.columns    
data_pca=pd.concat([data_pca,X_pca_project],axis=1)

for var in features:
    del data_pca[var]

i=0
combinaison_lineaire=pd.DataFrame(features.T,columns=["variable"])
for axe in liste_axes_retenus:
    comb=pd.DataFrame(pca.components_[i],columns=[axe])
    combinaison_lineaire=pd.concat([combinaison_lineaire,comb],axis=1)
    i=i+1
print(combinaison_lineaire)

#on crée une variable dichotomique par tag

df_tags_matrix= pd.DataFrame.from_dict(data_dict2, orient='index')
df_tags_matrix=df_tags_matrix[['Tags']]
df_tags_matrix['question']=df_tags_matrix.index
list_tag_brute=[]
for question in data_dict2:
    list_tag_brute=list_tag_brute+data_dict2[question]['Tags']
list_tag_brute=list_unique(list_tag_brute)

df_tags_matrix['Tags_str'] = [' '.join(map(str, l)) for l in df_tags_matrix['Tags']]
df_tags_matrix['Tags_str']=df_tags_matrix['Tags_str']+" "
for value in list_tag_brute:
    if value=="c++":
        df_tags_matrix["tag_"+value]=df_tags_matrix.Tags_str.str.contains("c\+\+")
        df_tags_matrix["tag_"+value] = df_tags_matrix["tag_"+value].astype(int)
    elif value=="java":
        df_tags_matrix["tag_"+value]=df_tags_matrix.Tags_str.str.contains("java ")
        df_tags_matrix["tag_"+value] = df_tags_matrix["tag_"+value].astype(int)

    else:
        df_tags_matrix["tag_"+value]=df_tags_matrix.Tags_str.str.contains(value)
        df_tags_matrix["tag_"+value] = df_tags_matrix["tag_"+value].astype(int)

data_pca_2=pd.concat([data_pca,df_tags_matrix],axis=1)

tag_acp_coordonnees = defaultdict(set)
for value in list_tag_brute:
    tag_acp_coordonnees[value]= defaultdict(set)
    for axe in liste_axes_retenus:
        tag_acp_coordonnees[value][axe]=""
        a=data_pca_2[data_pca_2["tag_"+value]==1][axe].mean()
        tag_acp_coordonnees[value][axe]=a
        
df_tags_acp_coordonnees= pd.DataFrame.from_dict(tag_acp_coordonnees, orient='index')
list_pca_strict=[]
for index, value in enumerate(list_tag_brute):
    if value=='python' or value=='java' or value=='pandas' or value=='javascript' or value=='.net-core':
        list_pca_strict.append(index)
df_tags_acp_coordonnees_sample=df_tags_acp_coordonnees.iloc[list_pca_strict]


fig, ax = plt.subplots()
ax.scatter(df_tags_acp_coordonnees['F1'], df_tags_acp_coordonnees['F2'])
ax.grid(True)
for i, txt in enumerate(df_tags_acp_coordonnees.index):
    ax.annotate(txt,(df_tags_acp_coordonnees['F1'][i],df_tags_acp_coordonnees['F2'][i]))
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()

fig, ax = plt.subplots()
ax.scatter(df_tags_acp_coordonnees_sample['F1'], df_tags_acp_coordonnees_sample['F2'])
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
for i, txt in enumerate(df_tags_acp_coordonnees_sample.index):
    ax.annotate(txt,(df_tags_acp_coordonnees_sample['F1'][i],df_tags_acp_coordonnees_sample['F2'][i]))
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()


fig, ax = plt.subplots()
ax.scatter(df_tags_acp_coordonnees['F3'], df_tags_acp_coordonnees['F4'])
ax.grid(True)

for i, txt in enumerate(df_tags_acp_coordonnees.index):
    ax.annotate(txt,(df_tags_acp_coordonnees['F3'][i],df_tags_acp_coordonnees['F4'][i]))
plt.xlabel("F3")
plt.ylabel("F4")
plt.show()

fig, ax = plt.subplots()
ax.scatter(df_tags_acp_coordonnees_sample['F3'], df_tags_acp_coordonnees_sample['F4'])
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
for i, txt in enumerate(df_tags_acp_coordonnees_sample.index):
    ax.annotate(txt,(df_tags_acp_coordonnees_sample['F3'][i],df_tags_acp_coordonnees_sample['F4'][i]))
plt.xlabel("F3")
plt.ylabel("F4")
plt.show()


###########################################################
#        MULTILAYERPERCEPTRON SUR LE TF ONE VS REST (9h)  #
###########################################################

print ("MULTILAYERPERCEPTRON SUR LE TF ONE VS REST")
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

tag_accuracy_score_apprentissage_mlp_tf=[]
tag_accuracy_score_generalisation_mlp_tf=[]
tag_best_param_mlp_tf=[]
df_tags_matrix_predict=df_tags_matrix['question']
df_tags_matrix_predict=df_tags_matrix_predict.reset_index(drop=True)
for tag in list_tag_brute:
    t1=time.time()
    X=tf_body
    y=df_tags_matrix[["tag_"+tag]].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    parameters = {'hidden_layer_sizes': [(10,),(10,10),(20,),(20,10),(20,10,10)],'solver':['lbfgs','adam']}
    clf2 = MLPClassifier(random_state=1, max_iter=2000)
    model = GridSearchCV(clf2, parameters,cv=5)
    clf=model.fit(X_train, y_train)    

    data_fname = 'mlp_tf_'+tag+'.pkl'
    with open(os.path.join(path_to_data, data_fname), "wb") as tf:
        pickle.dump(model,tf)

    data_fname = 'mlp_tf_'+tag+'.pkl'
    with open(os.path.join(path_to_data, data_fname), "rb") as tf:
        model = pickle.load(tf)  
    

    t2=time.time()
    print ("le mlp prend {} min à tourner".format((t2-t1)/60))
    y_predict=clf.predict_proba(X)
    temp=pd.DataFrame(y_predict[:,1], columns=["prob_prediction_1_mlp_tf_"+tag] )
    df_tags_matrix_predict=pd.concat([df_tags_matrix_predict,temp],axis=1)
    #score d apprentissage
    test_score=clf.score(X_train, y_train)
    temp2=(tag, test_score)
    tag_accuracy_score_apprentissage_mlp_tf.append(temp2)
    #score de généralisation
    test_score=clf.score(X_test, y_test)
    temp2=(tag, test_score)
    tag_accuracy_score_generalisation_mlp_tf.append(temp2)
    temp2=(tag, clf.best_params_)
    tag_best_param_mlp_tf.append(temp2)


###########################################################
#        MULTILAYERPERCEPTRON SUR LE TF MULTICLASS (9h)   #
###########################################################
print ("MULTILAYERPERCEPTRON SUR LE TF MULTICLASS")
#list_tag_brute=['java','javascript','python']
list_var_tags=[]
for value in list_tag_brute:
    list_var_tags.append("tag_"+value)

df_tags_matrix_multiclass_mlp_predict=df_tags_matrix['question']
df_tags_matrix_multiclass_mlp_predict=df_tags_matrix_multiclass_mlp_predict.reset_index(drop=True)

tag_accuracy_score_apprentissage_mlp_tf_multiclass=[]
tag_accuracy_score_generalisation_mlp_tf_multiclass=[]
tag_accuracy_score_apprentissage_mlp_tf_multiclass_detail=[]
tag_accuracy_score_generalisation_mlp_tf_multiclass_detail=[]
tag_best_param_mlp_tf_multiclass=[]

t1=time.time()
X=tf_body
y=df_tags_matrix[list_var_tags].values
y_strate=df_tags_matrix[['tag_python','tag_javascript']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y_strate, random_state=1)
parameters = {'hidden_layer_sizes': [(100,),(100,100),(200,100),(200,100,100)],'solver':['lbfgs','adam']}
clf2 = MLPClassifier(random_state=1, max_iter=2000)
model = GridSearchCV(clf2, parameters,cv=5)
clf=model.fit(X_train, y_train)  
#on sauvegarde

data_fname = 'mlp_tf_multiclass.pkl'
with open(os.path.join(path_to_data, data_fname), "wb") as tf:
    pickle.dump(model,tf)

data_fname = 'mlp_tf_multiclass.pkl'
with open(os.path.join(path_to_data, data_fname), "rb") as tf:
    model = pickle.load(tf)  

  
t2=time.time()
print ("le mlp prend {} min à tourner".format((t2-t1)/60))
y_predict=clf.predict_proba(X)
temp=pd.DataFrame(y_predict,  columns=["prob_prediction_1_mlp_multiclass_tf_"+i for i in list_var_tags] )
df_tags_matrix_multiclass_mlp_predict=pd.concat([df_tags_matrix_multiclass_mlp_predict,temp],axis=1)

#score d apprentissage

#test_score=clf.score(X_train, y_train)
#tag_accuracy_score_apprentissage_mlp_tf_multiclass.append(test_score)
pred=clf.predict(X_train)
test_score=accuracy_score(y_train,pred)
tag_accuracy_score_apprentissage_mlp_tf_multiclass.append(test_score)
x=0
while x<len(list_var_tags):
    temp=pd.DataFrame(pred[:,x],  columns=["prediction_mlp_multiclass_tf_"+list_var_tags[x]] )
    test_score=accuracy_score(y_train[:,x],temp["prediction_mlp_multiclass_tf_"+list_var_tags[x]])
    ts=(list_var_tags[x],test_score)
    tag_accuracy_score_apprentissage_mlp_tf_multiclass_detail.append(ts)
    x=x+1

#score de généralisation
#test_score=clf.score(X_test, y_test)
#tag_accuracy_score_generalisation_mlp_tf_multiclass.append(test_score)
pred=clf.predict(X_test)
test_score=accuracy_score(y_test,pred)
tag_accuracy_score_generalisation_mlp_tf_multiclass.append(test_score)
x=0
while x<len(list_var_tags):
    temp=pd.DataFrame(pred[:,x],  columns=["prediction_mlp_multiclass_tf_"+list_var_tags[x]] )
    test_score=accuracy_score(y_test[:,x],temp["prediction_mlp_multiclass_tf_"+list_var_tags[x]])
    ts=(list_var_tags[x],test_score)
    tag_accuracy_score_generalisation_mlp_tf_multiclass_detail.append(ts)
    x=x+1


tag_best_param_mlp_tf_multiclass.append(clf.best_params_)
    
###########################################################
#           MULTILAYERPERCEPTRON SUR LA MATRICE 10    #
###########################################################
# #tag_best_param_mlp_matrix=[]    
# tag_accuracy_score_mlp_matrix=[]
# for tag in list_tag_brute:
#     t1=time.time()
#     X=data_tsne_matrix
#     y=df_tags_matrix[["tag_"+tag]].values.ravel()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# #    parameters = {'alpha': 10.0 ** -np.arange(1, 10)}
#     model = MLPClassifier(random_state=1, max_iter=300)
# #    model = GridSearchCV(clf2, parameters,cv=5)
#     clf=model.fit(X_train, y_train)
#     t2=time.time()
#     print ("le mlp prend {} min à tourner".format((t2-t1)/60))
#     y_predict=clf.predict_proba(X)
#     temp=pd.DataFrame(y_predict[:,1], columns=["prob_prediction_1_mlp_matrix_"+tag] )
#     df_tags_matrix=pd.concat([df_tags_matrix,temp],axis=1)
#     test_score=clf.score(X_test, y_test)
#     temp2=(tag, test_score)
#     tag_accuracy_score_mlp_matrix.append(temp2)
# #    temp2=(tag, clf.best_params_)
# #    tag_best_param_mlp_matrix.append(temp2)


###########################################################
#           RANDOM FOREST SUR LE TF MULTICLASS            #
###########################################################


from sklearn.ensemble import RandomForestClassifier

df_tags_matrix_multiclass_rf_predict=df_tags_matrix['question']
df_tags_matrix_multiclass_rf_predict=df_tags_matrix_multiclass_rf_predict.reset_index(drop=True)

tag_accuracy_score_apprentissage_rf_tf_multiclass=[]
tag_accuracy_score_generalisation_rf_tf_multiclass=[]
tag_accuracy_score_apprentissage_rf_tf_multiclass_detail=[]
tag_accuracy_score_generalisation_rf_tf_multiclass_detail=[]
tag_best_param_rf_tf_multiclass=[]

t1=time.time()
X=tf_body
y=df_tags_matrix[list_var_tags].values
y_strate=df_tags_matrix[['tag_python','tag_javascript']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y_strate, random_state=1)
param_grid = { 
    'n_estimators': [100, 300, 500],
    'max_depth' : [None,3,4],
    'oob_score' : [True]
    }
model = RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
rfc=CV_rfc.fit(X_train, y_train)


data_fname = 'rf_tf_multiclass.pkl'
with open(os.path.join(path_to_data, data_fname), "wb") as tf:
    pickle.dump(CV_rfc,tf)

data_fname = 'rf_tf_multiclass.pkl'
with open(os.path.join(path_to_data, data_fname), "rb") as tf:
    CV_rfc = pickle.load(tf)  

t2=time.time()
print ("la rf prend {} min à tourner".format((t2-t1)/60))
y_predict=rfc.predict_proba(X)
x=0
while x<len(list_var_tags):
    list_temp=y_predict[x]
    temp=pd.DataFrame(list_temp[:,1],  columns=["prob_prediction_1_rf_multiclass_tf_"+list_var_tags[x]] )
    df_tags_matrix_multiclass_rf_predict=pd.concat([df_tags_matrix_multiclass_rf_predict,temp],axis=1)
    x=x+1
    
#score d apprentissage
pred=rfc.predict(X_train)
test_score=accuracy_score(y_train,pred)
tag_accuracy_score_apprentissage_rf_tf_multiclass.append(test_score)
x=0
while x<len(list_var_tags):
    temp=pd.DataFrame(pred[:,x],  columns=["prediction_rf_multiclass_tf_"+list_var_tags[x]] )
    test_score=accuracy_score(y_train[:,x],temp["prediction_rf_multiclass_tf_"+list_var_tags[x]])
    ts=(list_var_tags[x],test_score)
    tag_accuracy_score_apprentissage_rf_tf_multiclass_detail.append(ts)
    x=x+1

#score de généralisation
pred=rfc.predict(X_test)
test_score=accuracy_score(y_test,pred)
tag_accuracy_score_generalisation_rf_tf_multiclass.append(test_score)
x=0
while x<len(list_var_tags):
    temp=pd.DataFrame(pred[:,x],  columns=["prediction_rf_multiclass_tf_"+list_var_tags[x]] )
    test_score=accuracy_score(y_test[:,x],temp["prediction_rf_multiclass_tf_"+list_var_tags[x]])
    ts=(list_var_tags[x],test_score)
    tag_accuracy_score_generalisation_rf_tf_multiclass_detail.append(ts)
    x=x+1


tag_best_param_rf_tf_multiclass.append(rfc.best_params_)

###########################################################
#           RANDOM FOREST SUR LE MATRICE 10               #
###########################################################
# tag_accuracy_score_rf_matrix=[]
# for tag in list_tag_brute:
#     t1=time.time()
#     X=data_tsne_matrix
#     y=df_tags_matrix[["tag_"+tag]].values.ravel()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# #    param_grid = { 
# #        'n_estimators': [200, 500],
# #        'max_depth' : [None,4,5,6,7,8],
# #        'oob_score' : [True]
# #        }
#     model = RandomForestClassifier()
# #    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#     rfc=model.fit(X_train, y_train)
#     t2=time.time()
#     print ("la rf prend {} min à tourner".format((t2-t1)/60))
#     pred = rfc.predict(X_test)
#     temp2= (tag,accuracy_score(y_test, pred))
#     tag_accuracy_score_rf_matrix.append(temp2)
#     y_predict=rfc.predict_proba(X)
#     temp=pd.DataFrame(y_predict[:,1], columns=["prob_prediction_1_rf_matrix_"+tag] )
#     df_tags_matrix=pd.concat([df_tags_matrix,temp],axis=1)
    
    
    


    
###########################################################
#                    LDA BODY 20 TOPICS                   #
###########################################################
from sklearn.feature_extraction.text import CountVectorizer
#max_features=nombre de mots que l'on va garder dans le vocabulaire
tf_vectorizer_body=CountVectorizer(max_df=0.95, min_df=2, max_features=5000,stop_words='english')
tf_body=tf_vectorizer_body.fit_transform(list_body_model)

#Sparsicity is nothing but the percentage of non-zero datapoints in the document-word matrix, that is data_vectorized.
#Since most cells in this matrix will be zero, I am interested in knowing what percentage of cells 
#contain non-zero values.

# Materialize the sparse data
tf_body_dense = tf_body.todense()
# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((tf_body_dense > 0).sum()/tf_body_dense.size)*100, "%")
    
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
t1=time.time()
search_params = {'n_components': [20], 'learning_decay': [.5, .7, .9]}
lda_20 = LatentDirichletAllocation(max_iter=5, 
             learning_method='online', 
             learning_offset=50.,
             random_state=0)
model_20 = GridSearchCV(lda_20, param_grid=search_params,cv=5)
model_20.fit(tf_body)
t2=time.time()
print ("la lda body prend {} min à tourner".format((t2-t1)/60))
lda_body_20_best_model = model_20.best_estimator_
print("Best Model's Params lda tf_body: ", model_20.best_params_)
lda_body_20_best_params=model_20.best_params_
print("Best Log Likelihood Score: ", model_20.best_score_)
no_top_words=20
feature_names=tf_vectorizer_body.get_feature_names()
nbr_topic_lda_20_retenu=lda_body_20_best_params['n_components']
data_topic_body_lda_20= pd.DataFrame(index=np.arange(nbr_topic_lda_20_retenu))
data_topic_body_lda_20['topic_feature_lda_'+str(nbr_topic_lda_20_retenu)]=""
for topic_idx, topic in enumerate(lda_body_20_best_model.components_):
#        print("Topic {}:".format(topic_idx))
#        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    data_topic_body_lda_20['topic_feature_lda_'+str(nbr_topic_lda_20_retenu)][topic_idx]=[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]



###########################################################
#                T SNE A PARTIR DE LA MATRICE 20          #
###########################################################

#on calcule le nombre de mot du topic dans la question    

for question in data_dict2:
    index_topic=0
    while index_topic<=nbr_topic_lda_20_retenu-1:
        a=0
        list_topic=data_topic_body_lda_20['topic_feature_lda_'+str(nbr_topic_lda_20_retenu)][index_topic]
        for element in list_topic:
            data_dict2[question]['Body_token2']=data_dict2[question]['Body_token'].split(" ")
            if element in data_dict2[question]['Body_token2']:     
                a=a+1
        data_dict2[question]['topic_20s_'+str(index_topic)]=a
        data_dict2[question]['topic_20s_'+str(index_topic)]=data_dict2[question]['topic_20s_'+str(index_topic)]/20
        index_topic=index_topic+1

df_total_matrix_20= pd.DataFrame.from_dict(data_dict2, orient='index')
liste_topic=[]
x=0
while x<=nbr_topic_lda_20_retenu-1:
    liste_topic.append("topic_20s_"+str(x))
    x=x+1
data_tsne_matrix_20=df_total_matrix_20[liste_topic]
df_total_matrix_20['question']=df_total_matrix_20.index

data_topic_body_lda_20['mean_topic']=0
x=0
while x<nbr_topic_lda_20_retenu:
    data_topic_body_lda_20.loc[x, 'mean_topic'] = data_tsne_matrix_20["topic_20s_"+str(x)].mean()
    x=x+1

temp=data_tsne_matrix_20.copy()
temp['nbr_topic']=0
x=0
while x<nbr_topic_lda_20_retenu:
    temp["drap_topic_20s_"+str(x)]=np.where( temp["topic_20s_"+str(x)] > 0.25, 1, 0)
    temp['nbr_topic']=temp['nbr_topic']+temp["drap_topic_20s_"+str(x)]
    x=x+1 

plt.figure()
plt.hist(temp['nbr_topic'],bins=10)
plt.title('distribution du nombre de topic par question')
plt.xlabel('nombre de topic par question')
plt.ylabel('nombre de questions')
plt.show()    

data3=pd.DataFrame(list_question_brute,columns = ['question'])
from sklearn import manifold
n_components = 2
#perplexities = list(range(0, 51, 5))
perplexities=[20,40]
for j, perplexity in enumerate(perplexities):
    print('j={}'.format(j))
    tsne = manifold.TSNE(n_components=n_components,random_state=1, perplexity=perplexity)
    a=time.time()
    result_temp = tsne.fit_transform(data_tsne_matrix_20,list_question_brute)
    b=time.time()
    print('temps pour fitter avec une perplexité de {0}: {1}'.format(perplexity,((b-a)/60)))
    temp=pd.DataFrame(result_temp, columns=["data_tsne_20_results_"+str(perplexity)+"_F"+str(i+1) for i in range(2)] )
    data3=pd.concat([data3,temp],axis=1)

df_total_matrix_20['Tags_str'] = [' '.join(map(str, l)) for l in df_total_matrix_20['Tags']]
df_total_matrix_20['Tags_str']=df_total_matrix_20['Tags_str']+" "
df_total_matrix_20["tag_java"]=df_total_matrix_20.Tags_str.str.contains("java ")
df_total_matrix_20["tag_python"]=df_total_matrix_20.Tags_str.str.contains("python ")
df_total_tsne_20_matrix=pd.merge(df_total_matrix_20,data3,how='left',on='question')


for j, perplexity in enumerate(perplexities):
    colors = ['c', 'b']
    
    plt.figure()
    non_java = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==False]\
                                    ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==False]\
                                    ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    java = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((java, non_java),
               ('java', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne 20 avec une perplexité="+str(perplexity))
    plt.show()


    colors = ['y', 'b']
    plt.figure()
    non_python = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==False]\
                                    ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                                    df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==False]\
                                    ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])
    python = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])

    plt.legend((python, non_python),
               ('python', 'autres'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne 20 avec une perplexité="+str(perplexity))
    plt.show()


    colors = ['c', 'y']
    plt.figure()
    java = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_java']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[0])
    python = plt.scatter(df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F1'], \
                        df_total_tsne_20_matrix[df_total_tsne_20_matrix['tag_python']==True]\
                        ['data_tsne_20_results_'+str(perplexity)+'_F2'], marker='x', color=colors[1])

    plt.legend((python, java),
               ('python', 'java'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("tsne 20 avec une perplexité="+str(perplexity))
    plt.show()



# choix du nombre de composantes à calculer
n_comp = nbr_topic_lda_20_retenu

# selection des colonnes à prendre en compte dans l'ACP
data_pca_20=data_tsne_matrix_20.copy()

# préparation des données pour l'ACP
#data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
X = data_pca_20.values
names = data_pca_20.index # ou data.index pour avoir les intitulés
features = data_pca_20.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca_20 = decomposition.PCA(n_components=n_comp)
pca_20.fit(X_scaled)

# Eboulis des valeurs propres
display_scree_plot(pca_20)

# Cercle des corrélations
pcs_20 = pca.components_
display_circles(pcs_20, n_comp, pca_20, [(0,1),(2,3),(4,5)], labels = np.array(features))
#,(2,3),(4,5)
# Projection des individus
X_pca_20_projected = pca_20.transform(X_scaled)
X_pca_20_project=pd.DataFrame(X_pca_20_projected, index=names,columns=["F"+str(i+1) for i in range(n_comp)] )
explained_variance_sum=0
for i in range(n_comp):
    explained_variance_sum=explained_variance_sum+100*pca_20.explained_variance_ratio_[i]
    if explained_variance_sum>90:
        del X_pca_20_project['F'+str(i+1)]
liste_axes_retenus=X_pca_20_project.columns    
data_pca_20=pd.concat([data_pca_20,X_pca_20_project],axis=1)

for var in features:
    del data_pca_20[var]

i=0
combinaison_lineaire_20=pd.DataFrame(features.T,columns=["variable"])
for axe in liste_axes_retenus:
    comb_20=pd.DataFrame(pca_20.components_[i],columns=[axe])
    combinaison_lineaire_20=pd.concat([combinaison_lineaire_20,comb_20],axis=1)
    i=i+1
print(combinaison_lineaire_20)


#on crée une variable dichotomique par tag
df_tags_matrix_20= pd.DataFrame.from_dict(data_dict2, orient='index')
df_tags_matrix_20=df_tags_matrix_20[['Tags']]
df_tags_matrix_20['question']=df_tags_matrix_20.index

list_tag_brute_20=[]
for question in data_dict2:
    list_tag_brute_20=list_tag_brute_20+data_dict2[question]['Tags']
list_tag_brute_20=list_unique(list_tag_brute_20)

df_tags_matrix_20['Tags_str'] = [' '.join(map(str, l)) for l in df_tags_matrix_20['Tags']]
df_tags_matrix_20['Tags_str'] =df_tags_matrix_20['Tags_str'] +" "
for value in list_tag_brute_20:
    if value=="c++":
        df_tags_matrix_20["tag_"+value]=df_tags_matrix_20.Tags_str.str.contains("c\+\+")
        df_tags_matrix_20["tag_"+value] = df_tags_matrix_20["tag_"+value].astype(int)
    elif value=="java":
        df_tags_matrix_20["tag_"+value]=df_tags_matrix_20.Tags_str.str.contains("java ")
        df_tags_matrix_20["tag_"+value] = df_tags_matrix_20["tag_"+value].astype(int)
    else:
        df_tags_matrix_20["tag_"+value]=df_tags_matrix_20.Tags_str.str.contains(value)
        df_tags_matrix_20["tag_"+value] = df_tags_matrix_20["tag_"+value].astype(int)

data_pca_20_2=pd.concat([data_pca_20,df_tags_matrix_20],axis=1)

tag_20_acp_coordonnees = defaultdict(set)
for value in list_tag_brute:
    tag_20_acp_coordonnees[value]= defaultdict(set)
    for axe in liste_axes_retenus:
        tag_20_acp_coordonnees[value][axe]=""
        a=data_pca_20_2[data_pca_20_2["tag_"+value]==1][axe].mean()
        tag_20_acp_coordonnees[value][axe]=a
        
df_tags_20_acp_coordonnees= pd.DataFrame.from_dict(tag_20_acp_coordonnees, orient='index')
list_pca_strict_20=[]
for index, value in enumerate(list_tag_brute):
    if value=='python' or value=='java' or value=='pandas' or value=='javascript' or value=='.net-core':
        list_pca_strict_20.append(index)
df_tags_20_acp_coordonnees_sample=df_tags_20_acp_coordonnees.iloc[list_pca_strict_20]

fig, ax = plt.subplots()
ax.scatter(df_tags_20_acp_coordonnees['F1'], df_tags_20_acp_coordonnees['F2'])
ax.grid(True)
for i, txt in enumerate(df_tags_20_acp_coordonnees.index):
    ax.annotate(txt,(df_tags_20_acp_coordonnees['F1'][i],df_tags_20_acp_coordonnees['F2'][i]))
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()


fig, ax = plt.subplots()
ax.scatter(df_tags_20_acp_coordonnees_sample['F1'], df_tags_20_acp_coordonnees_sample['F2'])
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
for i, txt in enumerate(df_tags_20_acp_coordonnees_sample.index):
    ax.annotate(txt,(df_tags_20_acp_coordonnees_sample['F1'][i],df_tags_20_acp_coordonnees_sample['F2'][i]))
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()


fig, ax = plt.subplots()
ax.scatter(df_tags_20_acp_coordonnees['F3'], df_tags_20_acp_coordonnees['F4'])
ax.grid(True)
for i, txt in enumerate(df_tags_20_acp_coordonnees.index):
    ax.annotate(txt,(df_tags_20_acp_coordonnees['F3'][i],df_tags_20_acp_coordonnees['F4'][i]))
plt.xlabel("F3")
plt.ylabel("F4")
plt.show()

fig, ax = plt.subplots()
ax.scatter(df_tags_20_acp_coordonnees_sample['F3'], df_tags_20_acp_coordonnees_sample['F4'])
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
for i, txt in enumerate(df_tags_20_acp_coordonnees_sample.index):
    ax.annotate(txt,(df_tags_20_acp_coordonnees_sample['F3'][i],df_tags_20_acp_coordonnees_sample['F4'][i]))
plt.xlabel("F3")
plt.ylabel("F4")
plt.show()

# =============================================================================
# 
# ###########################################################
# #           MULTILAYERPERCEPTRON SUR LA MATRICE 20        #
# ###########################################################
# 
# tag_accuracy_score_apprentissage_mlp_matrix_20=[]
# tag_accuracy_score_generalisation_mlp_matrix_20=[]
# tag_best_param_mlp_matrix_20=[]
# 
# for tag in list_tag_brute_20:
#     t1=time.time()
#     X=data_tsne_matrix_20
#     y=df_tags_matrix_20[["tag_"+tag]].values.ravel()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
#     parameters = {'hidden_layer_sizes': [(10,),(10,10),(20,10)]}
#     clf2 = MLPClassifier(random_state=1, max_iter=300)
#     model = GridSearchCV(clf2, parameters,cv=5)
#     clf=model.fit(X_train, y_train)
#     t2=time.time()
#     print ("le mlp prend {} min à tourner".format((t2-t1)/60))
#     y_predict=clf.predict_proba(X)
#     temp=pd.DataFrame(y_predict[:,1], columns=["prob_prediction_1_mlp_matrix_20_"+tag] )
#     data=pd.concat([df_tags_matrix_20,temp],axis=1)
#     #score d apprentissage
#     test_score=clf.score(X_train, y_train)
#     temp2=(tag, test_score)
#     tag_accuracy_score_apprentissage_mlp_matrix_20.append(temp2)
#     #score de généralisation
#     test_score=clf.score(X_test, y_test)
#     temp2=(tag, test_score)
#     tag_accuracy_score_generalisation_mlp_matrix_20.append(temp2)
#     temp2=(tag, clf.best_params_)
#     tag_best_param_mlp_matrix_20.append(temp2)
# 
# 
# ###########################################################
# #               RANDOM FOREST SUR LA MATRICE 20           #
# ###########################################################
# 
# tag_accuracy_score_rf_matrix_20=[]
# for tag in list_tag_brute_20:
#     t1=time.time()
#     X=data_tsne_matrix_20
#     y=df_tags_matrix_20[["tag_"+tag]].values.ravel()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# #    param_grid = { 
# #        'n_estimators': [200, 500],
# #        'max_depth' : [None,4,5,6,7,8],
# #        'oob_score' : [True]
# #        }
#     model = RandomForestClassifier()
# #    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#     rfc=model.fit(X_train, y_train)
#     t2=time.time()
#     print ("la rf prend {} min à tourner".format((t2-t1)/60))
#     pred = rfc.predict(X_test)
#     temp2= (tag,accuracy_score(y_test, pred))
#     tag_accuracy_score_rf_matrix_20.append(temp2)
#     y_predict=rfc.predict_proba(X)
#     temp=pd.DataFrame(y_predict[:,1], columns=["prob_prediction_1_rf_matrix_20_"+tag] )
#     data=pd.concat([df_tags_matrix_20,temp],axis=1)
# 
# =============================================================================
