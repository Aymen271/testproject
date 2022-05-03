from django.shortcuts import render
from django.http import HttpResponse
import re
from django.shortcuts import render, redirect
import PyPDF2
from PyPDF2 import PdfFileWriter, PdfFileReader
import pandas as pd  
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, HttpResponseNotFound
import json
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from ast import literal_eval

from .models import Todo
from .forms import TodoForm

import pdfplumber
import PyPDF2
import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer 
from strsimpy.jaro_winkler import JaroWinkler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder 
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
from collections import namedtuple
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, brown, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer , CountVectorizer 
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy 
import en_core_web_sm
import rdflib
from rdflib import Namespace
from owlready2 import *
import os
import spacy

import numpy as np
import pandas as pd
from strsimpy.jaro_winkler import JaroWinkler

nlp = en_core_web_sm.load()

def home(request):

    # template = loader.get_template('index2.html')
    # var="said"
    # return HttpResponse(template.render(request))
    return render(request,'index2.html',context)





reference = '\n A Guide to the Project Management Body of Knowledge (PMBOK\n® Guide) Œ Fifth Edition'
licence = 'Licensed To: Jorge Diego Fuentes Sanchez PMI MemberID: 2399412This copy is a PMI Member benefit, not for distribution, sale, or reproduction.'
clean = '© project management institute. a guide to the project management body of knowledge pmbok® guide – fifth edition . licensed to jorge diego fuentes sanchez pmi memberid'
clean2 = 'this copy is a pmi member benefit not for distribution sale or reproduction'
clean3="insignificant cost cost – cost – cost cost. cost. increase increase increase increase increase. insignificant time time – time – time time. time. increase increase increase increase increase."
s = '© project management institute. a guide to the project management body of knowledge pmbok® guide – fifth edition licensed to jorge diego fuentes sanchez pmi memberid .'
def cleaning(text):
    text=text.replace(reference,'')
    text=text.replace(licence,'')
    for i in range (313,355):
        text=text.replace(str(i)+'©2013 Project Management Institute.','')
    text=text.replace('1111.',' ')
    text=text.replace('11.',' 11.')
    text=text.replace('˜',' ').replace('•','')
    text = text.replace("\nem", " ").replace("™", "'").replace("˚"," ").replace("œ", " ").replace("š"," ")
    text=text.replace('\n',' ') 
    text=text.lower()
    text=text.replace('human','human ').replace('management',' management ').replace('described',' described').replace('estimates','estimate').replace('estimate',' estimate ').replace('identify',' identify ').replace('techniques',' techniques ').replace('risks','risk').replace('risk',' risk ')
    text= text.translate(str.maketrans('', '', '!"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'))
    text = re.sub(r'[*\d]', '', text)
    text = re.sub(' +', ' ', text)
    text = text.replace(clean,'').replace(clean2,'').replace('identify ing','identifying').replace(clean3,'').replace('such as.','such as')
    return text


stopWords = stopwords.words('english')
sw = set(stopWords) -set([ "it's", 'who', 'whom', 'is','in','of','for','are', 'was', 'were', 'be',
    'being', 'have', 'has','an', 'had', 'having', 'do', 'does', 'did', 'doing',"a", 'other','or', 'some',  'no', 'not', 'own', 'same', 'can', 'don', "don't", 'should',
     'now','will','such','as','can'])
sw.add('the')
sw.add('that')

stopWords.append(['the','that'])

def split_sentences(text):  
    text = re.split('[.?]', text)
    return text

def sentence_preprocessing(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words if not word in sw]
    return words


def data_split(corpus) :
    data = []
    process = split_sentences(corpus)
    for sentence in process :
        if sentence  :
            l = sentence_preprocessing(sentence)
            if l :
                data.append(l) # word tokenization for each sentence
    return data


#input liste de liste de tokens
def pos_tagging_nltk(tokens):
    return [nltk.pos_tag(w) for w in tokens if w]

def lemmatization(sent): 
    lemmatizer = WordNetLemmatizer()
    l = []
    for word, tag in sent :
        if word not in ['wbs','as'] :
            l.append((lemmatizer.lemmatize(word),tag))
        else :
            l.append((word,tag))

    return l


# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)



def process_sentences_after_cleaning(corpus):
    data=data_split(corpus)
    tagged = pos_tagging_nltk(data)
    lemm_tag = [lemmatization(sentence) for sentence in tagged]
    ch=""
    for l in lemm_tag :
        for w,k in l :
            ch=ch+" "+ w
        ch+='.'
    return ch



def lem_list(p1_inputs) :
    p1_inputsL=[]
    for l3 in p1_inputs :
        p1_inputsL.append(l3.lower().split(" "))

    
    p1_inputsLem=[]
    for l4 in p1_inputsL :
        tagged1 = nltk.pos_tag(l4) 
        #print(tagged1)
        lem = lemmatization(tagged1)
        s=''
        for j in lem :
            s=s+j[0]+' '

        p1_inputsLem.append(s[:-1])
    return p1_inputsLem

def split_concept(sentence):
    liste = []
    mots=sentence.split(' ')
    tagged =nltk.pos_tag(mots)
    
    ch=""
    for word, tag in tagged:
        if (tag[0] == 'N'):
            ch += word+' '
       
    l = ch.split(' ')
    
    ch2=''
    for i in range(len(l)-2,-1,-1) : 
        ch2=l[i]
        if (ch2[0:len(ch)-1] != sentence):
                liste.append(ch2)
        
        j=i-1
        while (j>-1):
            ch2=l[j]+' '+ch2
            #print(ch2)
            if (ch2[0:len(ch)-1] != sentence):
                liste.append(ch2)
            j-=1
        ch2=''


    return liste


######chunking
def list_chnk(lemm_tag) :

    list_grammar=[
                  "Chunk:{<NN.?><JJ.?><NN.?>}",
                  "Chunk:{<NN.?><NN.?><NN.?>}",
                  "Chunk:{<NN.?><NN.?>}",
                  "Chunk:{<NN.?>}",
                 
                  
                  
                  "Chunk:{<JJ.?><NN.?>}",
        
                  "Chunk:{<NN.?><VB.?><JJ>*<NN.?>}",
                  "Chunk:{<JJ.?><NN.?><NN.?><NN.?>}",
                  "Chunk:{<JJ.?><NN.?><NN.?>}"
  
                 ]
    list_grammar2=["Chunk:{<NN.?>}",
                
               
                "Chunk:{<NN.?><NN.?>}",
                 
                  
                  
                "Chunk:{<JJ.?><NN.?>}",
        
                "Chunk:{<NN.?><VB.?><JJ>*<NN.?>}",
                "Chunk:{<JJ.?><NN.?><NN.?><NN.?>}",
                "Chunk:{<JJ.?><NN.?><NN.?>}"
  
                ]
    
    list_chunked = []
    for k in list_grammar :
        chunkParser = nltk.RegexpParser(k)

        for tags in lemm_tag:
            chunked = chunkParser.parse(tags)
            list_chunked.append(chunked)
    for h in list_grammar2 :
        chunkParser = nltk.RegexpParser(h)

        for tags in lemm_tag:
            chunked = chunkParser.parse(tags)
            if chunked not in list_chunked :
               list_chunked.append(chunked)
    return list_chunked 


def liste_treef(list_chunked):
    liste_tree=[]
    for chunked in list_chunked :
        tree=[]
        try:
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                tree.append(subtree)
        except Exception as e:
            print(str(e))
            
        if tree :
            liste_tree = liste_tree + tree
    return liste_tree    



from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

def preprocess(corpus):
    data=data_split(corpus)
    tagged = pos_tagging_nltk(data)
    ##tagged = nltk.word_tokenize(data)
    lemm_tag = [lemmatization(sentence) for sentence in tagged]
    ##lemm_tag = [lemmatizer.lemmatize(sentence) for sentence in tagged]
    ###lemm_tag = lemmatize_sentence(corpus) 
    print(lemm_tag)
    list_chunked=list_chnk(lemm_tag)
    liste_tree=liste_treef(list_chunked)
    obj=obj_df (liste_tree)
    obj = obj.drop_duplicates().reset_index(drop=True)

    return obj  



def obj_df (liste_tree):
    Concepts=[]
    relations=[]
    objects=[]
    print(len(liste_tree))
    for i in range(len(liste_tree)-1):
        concept = ''
        v = False
        rel = ''
        obje = ''
        for term, tag in liste_tree[i] :
          
            if ( ( (tag[0]=='N') or (tag[0] == 'J')) and ( v == False )  ):
                concept = concept + term + ' '
              
            elif (tag[0] == 'V')  :
                v = True
                rel = rel + term + ' '

              
            elif  ((tag[0]=='N') and ( v == True ) ) :
                obje = obje + term + ' '


        Concepts.append(concept[:-1])           
        relations.append(rel[:-1])
        objects.append(obje[:-1])

    obj= pd.DataFrame(columns=['Concepts'])
    obj['Concepts']=Concepts
    return obj


pattern1N = [
             [  {'DEP':'amod','OP':"?"}, 
                {'POS':'NOUN','OP':"+"},
                {'POS':'NOUN','OP':"+"},
                {'POS': 'NOUN','OP':"+"},
                 {'POS':'VERB','OP':"+"}],
              [  {'DEP':'amod','OP':"?"}, 
              {'POS':'VERB','OP':"+"},
              {'POS': 'NOUN','OP':"+"}],
             #pattern 5
             [  {'DEP':'amod','OP':"?"}, 
                {'POS':'NOUN','OP':"+"},
                {'LOWER': 'and'},
                {'POS': 'NOUN','OP':"+"},
           {'POS':'NOUN','OP':"+"}],
             
    #pattern 6 
             [  {'DEP':'amod','OP':"?"}, 
                {'POS':'NOUN','OP':"+"},
              {'POS':'NOUN','OP':"+"},
                {'POS': 'NOUN','OP':"+"},
           {'POS':'NOUN','OP':"+"}],
    
             #pattern 7
             [  {'DEP':'amod','OP':"?"}, 
                {'POS':'NOUN','OP':"+"},
                {'LOWER': 'for'},
                {'POS':'ADJ','OP':"+"},
                {'POS': 'NOUN','OP':"+"},
                {'LOWER': 'or'},
                {'POS':'NOUN','OP':"+"}],
    
              #pattern 8
             [  {'DEP':'amod','OP':"?"}, 
                {'POS':'VERB','OP':"+"},
                {'POS':'ADJ','OP':"+"},
                {'POS': 'NOUN','OP':"+"},
                {'POS':'NOUN','OP':"+"}],
             
            ]


def matchN(doc):
    matcher = Matcher(nlp.vocab)
    matcher.add('All match', pattern1N)
    matchess = matcher(doc)
    spans = [doc[start:end] for (_, start, end) in matchess]
    return spans


def spacy_matcher_concepts(corpus):
    docN=nlp(corpus)
    spansN =matchN(docN)
    LSpans = []
    for l in spansN:
        LSpans.append(l.text)
    LSpansLem = lem_list(LSpans)
    return LSpansLem



def filtre(liste):
    l=[]
    for sent in liste :
        if (len(sent)>3):

            ch=''
            for w in sent.split(' '):
                if ((len(w)>1) and (len(w)<15)):
                    ch=ch+' '+w
            l.append(ch[1:])
    return l 

def get_concept(l1,l2):
    l= l1 + l2
    return (filtre(list(set(l))))


def spli_jnt (list001) :
  
 
    L=[]
    for s in list001 :
        listOfList = []
        listOfList = s.split(" ")
        listOfList.append(listOfList)
        listOfList.pop()
        L.append(listOfList)
    return L 








##########################Les fonction system de recommendation ################


def test_sim(user_input,chosen_column,df_process):
  jarowinkler = JaroWinkler()  
  merge=pd.DataFrame(columns=['Concept','Type', 'Definition','Process_Name','Similarite'])
  compteur= 0
  for i in chosen_column:
    compteur= compteur + 1
    for j in user_input:
      if (jarowinkler.similarity(i,j) > 0.75):
        #print(jarowinkler.similarity(i,j))
        merge=merge.append(df_process.loc[compteur-1])
        merge.at[compteur-1,'Similarite']=jarowinkler.similarity(i,j)
        
  merge=merge.sort_values(by = 'Similarite',ascending = False)
  return merge



def get_process(lis):
  for i in range(0,lis.shape[0]) :
    if (lis.iloc[i]['Type']=='Process') and (lis.iloc[i]['Similarite']==1):
      return lis.iloc[i]['Concept']



def get_user_input_type(liste):
  jarowinkler = JaroWinkler()
  
  type_list=[]
  for i in liste:
    if (jarowinkler.similarity("input",i)> 0.75):
      type_list.append("input")
    if (jarowinkler.similarity("output",i)> 0.75):
      type_list.append("output")
    if (jarowinkler.similarity("tools techniques",i)> 0.75):
      type_list.append("tools and techniques")
  return type_list

def showOne(input_user,df_sim,df_process):
  #case where we only have process in user list
  response=pd.DataFrame(columns=['Concept','Type', 'Definition','Process_Name','Similarite'])
  type_list=get_user_input_type(input_user)
  # ken je process wahdo
  if df_sim.iloc[0]["Type"] == "Process":
    if not type_list:
      type_list.append("tools and techniques")
      type_list.append("output")
      type_list.append("input")

    for i in type_list:
      for j in range(0,df_process.shape[0]) :
        if (df_process.iloc[j]['Type']==i) and (df_process.iloc[j]['Process_Name'] == get_process(df_sim)):
          response=response.append(df_process.iloc[j])
    response= response.append(df_sim.iloc[0])
  
  #ken je concept + process  
  elif df_sim.iloc[0]["Type"] != "Process":
    if get_process(df_sim):
      for j in range(0,df_sim.shape[0]) :
        if (df_sim.iloc[j]['Similarite']==1) and (df_sim.iloc[j]['Process_Name'] == get_process(df_sim)):
          response=response.append(df_sim.iloc[j])  
  # ken je concept wahdo
    else:
      if type_list:
        for i in get_user_input_type(input_user):
          for j in range(0,df_process.shape[0]) :
            if (df_process.iloc[j]['Type']==i) and (df_process.iloc[j]['Concept'] == df_sim.iloc[0]["Concept"] ):
              response=response.append(df_process.iloc[j])
      else: 
        response=response.append(df_sim.iloc[0]) 
        for j in range(0,df_process.shape[0]) :
            if (df_process.iloc[j]['Concept'] == df_sim.iloc[0]["Process_Name"] ):
              response=response.append(df_process.iloc[j])

    
  return response

  #############################################################################



def index(request):
    todo_list = Todo.objects.order_by('id')

    form = TodoForm()

    context = {'todo_list' : todo_list, 'form' : form}

    return render(request, 'index2.html', context)

@require_POST
def addTodo(request):



    form = TodoForm(request.POST)

    requette =''
    if form.is_valid():
        requette = Todo(text=request.POST['text'])


    requette = str(requette)   
    path="."
    new_df = pd.read_csv("dfontology.csv",sep=",")
    list_key = new_df['Concept'].tolist()
    list_value = new_df['Synonyms'].tolist()

    new_ll = []
    for elt in list_value:
        new_ll.append(literal_eval(elt))


    for i in range (0,len(new_ll)):
        if new_ll[i] == None:
            new_ll[i] = ['0']
    new_ll

    zip_iterator = zip(list_key, new_ll)
    bigD = dict(zip_iterator)


    requette = requette.lower()
    l_rq = requette.split(" ")
    rq = cleaning(str(requette))




    form = TodoForm()



    newcase = rq.split(' ')

    v = True
    for e in ['processes','project','risk','management']  :
        if e not in newcase :
            v = False
            break

    if v :
        dfp = new_df[new_df.Type=='Process']

        processes = json.loads(dfp.reset_index().to_json(orient ='records'))
        return render(request, 'index2.html', {"processes":processes,'form':form})



    conc_user = preprocess(rq)
    spy = nlp(requette)
    conc_user_spy=spacy_matcher_concepts(requette)
    Concepts_process_user  =  get_concept(list(conc_user['Concepts']),conc_user_spy)
    
    conc_user_list_filtre = []
    conc_user_list_filtre = get_user_input_type(l_rq)

    #Ld = ['input', 'output','technique', 'tool','technique tool','process','outputs','inputs','techniques', 'tools','techniques tools']

    

    df_concepts_process_user = pd.DataFrame(Concepts_process_user,columns =['Concepts'])
    df_concepts_process_user = df_concepts_process_user.drop_duplicates()
    conc_user_list = df_concepts_process_user["Concepts"].tolist()

    k = 0

    for i in df_concepts_process_user['Concepts'].tolist():
        for key, value in bigD.items():
        
            if (i not in list_key):
                if (i in value) :
                    print("i ",i, " value ", value)
                    conc_user_list[k]= key 
                    for item in conc_user_list: 
                        p = conc_user_list.index(item)
                        conc_user_list[p]= item.replace(i,key) 
        k+=1
    
    list_concpt01=new_df['Concept'].tolist()
    for i in list_concpt01 :
       if (rq.find(i)==0) :
           conc_user_list.append(i)
    list(set(conc_user_list))


    for i in range(0,len(conc_user_list)):
         for j in range(i+1,len(conc_user_list)):
                if (conc_user_list[i].find(conc_user_list[j])) != -1 :
                    conc_user_list[j] = conc_user_list[i]
                if (conc_user_list[j].find(conc_user_list[i])) != -1:
                    conc_user_list[i] = conc_user_list[j]
    conc_user_list = list(set(conc_user_list)) 

    for i in range(0,len(conc_user_list)):
        conc_user_list[i] = conc_user_list[i].replace(' input','')
        conc_user_list[i] = conc_user_list[i].replace(' output','')
        conc_user_list[i] = conc_user_list[i].replace(' tool','')
        conc_user_list[i] = conc_user_list[i].replace(' technique','')
        conc_user_list[i] = conc_user_list[i].replace(' technique tool','')
        conc_user_list[i] = conc_user_list[i].replace(' process','')


    l_all_c = []
    for key, value in bigD.items():
        l_all_c.append(key)
    for l in conc_user_list:
        if l in l_all_c:
            conc_user_list_filtre.append(l)
    

    conc_user_list_filtre= list(set(conc_user_list_filtre))





####################################Recommendation###################################
      
    df_process = pd.read_csv("dfontology.csv",sep=",")

   

    df_process['Type'] = df_process['Type'].replace(['Has_inputs'],'input')
    df_process['Type'] = df_process['Type'].replace(['Has_Technics_Tools'],'tools and techniques')
    df_process['Type'] = df_process['Type'].replace(['Has_outputs'],'output')

   
    #process kahaw
   # input_user= ["identify risk","input"]
    #concept kahaw
    #input_user= ["risk register"]
    #input_user=[""]
    #if input feha process + esm chapitre donc affiche les 6 process
    #process + concept
    #input_user= ["risk register","identify risk"]

    #input_user= ["risk register","identify risk"]
    input_user = conc_user_list_filtre
    
    jarowinkler = JaroWinkler()

    concept_sim_df= df_process['Concept']

    df_sim=test_sim(input_user,concept_sim_df,df_process)

    if not input_user or df_sim.shape[0] == 0:
        return render(request, 'index2.html', {"message":"Please provide more details !",'form':form})

    #df_sim

    #exemple appel
    result = showOne(input_user,df_sim,df_process)

    df1 = result[result.Type=='Process']
    df2 = result[result.Type=='input']
    df3 = result[result.Type=='tools and techniques']
    df4 = result[result.Type=='output']


    df11 = json.loads(df1.reset_index().to_json(orient ='records'))
    df22 = json.loads(df2.reset_index().to_json(orient ='records'))
    df33 = json.loads(df3.reset_index().to_json(orient ='records'))
    df44 = json.loads(df4.reset_index().to_json(orient ='records'))


    

    lsys = df1['Synonyms'].tolist()
    lsyno = lsys[0]
    lsyno = lsyno[2:len(lsyno)-2].split("', '")


   
   
    print(lsyno)
    SynonymsList = json.loads(pd.DataFrame(columns=["Synonyms"],data=lsyno).drop_duplicates().reset_index().to_json(orient ='records'))
    context2= { 'form' : form, 'df11': df11, 'df22': df22, 'df33': df33, 'df44': df44 ,'lsynonyms':SynonymsList,'message':''} 


    return render(request, 'index2.html', context2)
    #return HttpResponse(result.shape[0])



##########################section pdf ####################################

# Create your views here.
def pmbokSection(request):
    concepts = ['Project charter', 'Stakeholder register', 'Enterprise Environmental Factors', 'organizational Process Assets', 'risk Management Plan', 'cost Management Plan', 'Schedule Management Plan', 'Quality Management Plan', 'Human resource Management Plan', 'Scope Baseline', 'Activity cost Estimates', 'Activity duration Estimates', 'Stakeholder register', 'Enterprise Environmental Factors', 'organizational Process Assets', 'risk Management Plan', 'Scope Baseline', 'risk register', 'Enterprise Environmental Factors', 'organizational Process Assets', 'risk Management Plan', 'cost Management Plan', 'Schedule Management Plan', 'risk register', 'Enterprise Environmental Factors', 'organizational Process Assets', 'Project Management Plan', 'Work Performance data', 'Work Performance reports']
    sections =['section 4.1.3.1.', 'section 13.1.3.1.', 'section 2.1.5.', 'section 2.1.4.', 'section 11.1.3.1.', 'section 7.1.3.1.', 'section 6.1.3.1.', 'section 8.1.3.1.', 'section 9.1.3.1.', 'section 5.4.3.1.', 'section 7.2.3.1.', 'section 6.5.3.1.', 'section 13.1.3.1.', 'section 2.1.5.', 'section 2.1.4.', 'section 11.1.3.1.', 'section 5.4.3.1.', 'section 11.2.3.1.', 'section 2.1.5.', 'section 2.1.4.', 'section 11.1.3.1.', 'section 7.1.3.1.', 'section 6.1.3.1.', 'section 11.2.3.1.', 'section 2.1.5.', 'section 2.1.4.', 'section 4.2.3.1.', 'section 4.3.3.2.', 'section 4.4.3.2.']
    data = pd.DataFrame({'concept':concepts,'range':sections}).drop_duplicates()
    json_records = data.reset_index().to_json(orient ='records')
    dataset = []
    dataset = json.loads(json_records)
    context = {'d': dataset} 
    return render(request, 'table.html', context)



def displaySection(section):
    object = PyPDF2.PdfFileReader("PMBOK.pdf")
    NumPages = object.getNumPages()
    #String = "section 11.2.3.1."
    p=0
    for i in range(0, NumPages):
        Text =object.getPage(i).extractText() 
        ResSearch = re.search(section[8:-1]+' ', Text)
        if (ResSearch) :
            p=i
            break
    
    return p

def Pages_To_Keep(section):
    output = PdfFileWriter()
    infile = PdfFileReader('PMBOK.pdf', 'rb')
    index=0
    list_index = []
    print(section)
    text=""
    search=False
    ResSearch=False
    ch=section[8:-1]+' '
    if len(ch) != 6 :
        for i in range(26, infile.getNumPages()):
            text =infile.getPage(i).extractText()
            num = 0
            for line in text.split('\n'):
                num+=1
                if num>3 :
                    ResSearch = re.search(ch, line)
                    if (ResSearch) :
                        list_index.append(i)
                        search=True

                    if search :
                        if re.search('^\d+.\d\s',line) or (line.startswith('13 - P')):
                            print(line)
                            search= False
                            break
                        if (re.search('[a-z].\d.\d\s[A-Z]',line)) or (re.search('[A-Z]\d+.\d\s[A-Z]',line)) :
                            search = False 
                            print(line)
                            print(i)
                            break
                        if( re.match('^(\d.)+[0-9]\s',line) and not (re.search(ch, line))):
                            print(line)
                            search = False
                            break
                            
                        if (ch[len(ch)-4:len(ch)-1] =='3.1') :
                            #print('here')
                            if (re.search('\d.\d.3.2\s',line)) :
                                print(line)
                                search = False
                                
                        if ((i not in list_index) and search) :
                            list_index.append(i)
    else :
        
        for i in range(26, infile.getNumPages()):
            text =infile.getPage(i).extractText()
            num =0
            for line in text.split('\n'):
                num+=1
                if num>4 :
                    if  line.startswith(ch) :
                        ResSearch =  True
                    elif re.search(ch, line):
                        if (re.search('[A-Za-b]\d.\d.\d\s',line)) :
                            ResSearch =  True
                    if (ResSearch) :
                        list_index.append(i)
                        search=True
                        ResSearch=False


                    if search :
                        if re.search('^\d+.\d\s',line) or (re.search('[a-z].\d.\d\s[A-Z]',line)) or ( re.match('^(\d.){2}[0-9]\s',line[0:6]) ):
                            if line[0:6] != ch:
                             #   print(line)
                                search = False  
                                break
                    

                        if ((i not in list_index) and search) :
                            list_index.append(i)
    
#    if len(list_index)> 3 :
#        list_index=(list_index[0:3])

    for index in list_index:
        p = infile.getPage(index)
        print(infile.getPage(index+1).extractText())
        output.addPage(p)
    filename='section/'+section.replace(' ','')+'pdf'

    with open(filename, 'wb') as f:
        output.write(f)
    return filename



    

def pdf_view(request, section):
    print(section +'****************')
    fs = FileSystemStorage()
    filename = Pages_To_Keep(section)
    if fs.exists(filename):
        with fs.open(filename) as pdf:
            response = HttpResponse(pdf, content_type='application/pdf')
            #response['Content-Disposition'] = 'attachment; filename="mypdf.pdf"' #user will be prompted with the browser’s open/save file
            response['Content-Disposition'] = 'inline; filename=filename' #user will be prompted display the PDF in the browser
            return response
    else:
        return HttpResponseNotFound('The requested pdf was not found in our server.')

'''
def pdf(request):
    try:
        return FileResponse(open('<file name with path>', 'rb'), content_type='application/pdf')
    except FileNotFoundError:
        raise Http404('not found')
'''

