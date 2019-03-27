#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import time
import string


# ## Import UCI-Sentiment Labelled Dataset 
# __[Click here to download](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)__

# ##### This data set contains sentences from 3 sources
# - *Amazon*
# - *Imdb*
# - *Yelp*

# In[2]:


amz=pd.read_csv("amazon.txt",delimiter="\t",header=None)
imdb=pd.read_csv("imdb.txt",delimiter="\t",header=None)
yelp=pd.read_csv("yelp.txt",delimiter="\t",header=None)


# In[3]:


columns=['Sentence','Review']


# In[4]:


_=amz.columns=columns
_=imdb.columns=columns
_=yelp.columns=columns


# In[5]:


amz.index=[i for i in range(1,amz.shape[0]+1)]
imdb.index=[i for i in range(1,imdb.shape[0]+1)]
yelp.index=[i for i in range(1,yelp.shape[0]+1)]


# ## Merge all sentences into single dataframe

# In[6]:


sent=pd.concat([amz,imdb,yelp])


# In[7]:


sent.reset_index(drop=True,inplace=True)


# In[8]:


from sklearn.utils import shuffle
sent=shuffle(sent)


# In[9]:


sent["Id"]=[i for i in range(1,sent.shape[0]+1)]


# In[10]:


sent.index=sent["Id"]
sent.drop(["Id"],axis=1,inplace=True)


# In[11]:


for i in range(1,sent.shape[0]+1):
    if sent.at[i,"Review"] == 0:
        sent.at[i,"Review"]=-1


# In[12]:


for i in range(1,sent.shape[0]+1):
    sent.at[i,"Sentence"]=sent.at[i,"Sentence"].strip().lower()


# ## Split dataset into Sentences and labels

# In[13]:


se=np.array(sent["Sentence"])
rev=np.array(sent["Review"])


# In[14]:


rev=rev.astype('int8')


# ## Preprocess data
# 1.Remove Digits <br>
# 2.Remove Punctuations <br>
# 3.Remvoe Stop Words

# In[15]:


punc_list=list(string.punctuation)


# In[16]:


stop_list=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# In[17]:


import re


# In[18]:


def rm_dig(l):
    ll=[]
    for i in range(len(l)):
        x=re.sub(r"\d+",' ',l[i])
        x=re.sub(r"\s+",' ',x)
        ll.append(x.strip())
    return ll


# In[19]:


se_digit_less=rm_dig(se)


# In[20]:


def rm_punc(l,rl):
    for w in rl:
        l=l.replace(w,' ')
        l=re.sub('\s+',' ',l)
    return l
se_punc_less=[rm_punc(l,punc_list) for l in se_digit_less ]


# In[21]:


def rm(l,rl):
    re=[]
    for i in range(len(l)):
        ll=l[i].strip().split(" ")
        lr=[i for i in ll if i not in rl]
        re.append(" ".join(lr))
    return re


# In[22]:


se_clear=rm(se_punc_less,stop_list)


# ## Apply Tf-Idf Vectorization on preprocessed data set

# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[24]:


vectorizer=TfidfVectorizer(analyzer="word",tokenizer=None,preprocessor=None,lowercase=False,stop_words=None,max_features=4500)


# In[25]:


X=vectorizer.fit_transform(se)


# In[26]:


X_mat=X.toarray()


# In[27]:


X_mat.shape


# ## Split dataset into training and testing sets

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(X_mat,rev,test_size=0.25)


# In[30]:


y_train=np.squeeze(y_train)
y_test=np.squeeze(y_test)


# In[31]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# ## Apply Linear SVM 

# In[32]:


from sklearn import svm


# In[33]:


def fit_classifier(c):
    clf_svc=svm.SVC(kernel='linear',gamma='scale',C=c)
    tb=time.time()
    clf_svc.fit(x_train,y_train)
    pred_svc=clf_svc.predict(x_test)
    tf=time.time()
    co=0
    for i,j in zip(pred_svc,y_test):
        if i==j:
            co+=1
    accuracy=co/float(len(y_test))
    t=tf-tb
    return [accuracy,t]


# In[34]:


c_values=[1.0,1.5,2.0]
for i,ele in enumerate(c_values):
    print("C=",c_values[i],end=' ')
    l=fit_classifier(ele)
    print("Accuracy=",l[0],' ',"Time=",l[1])


# For differenct values of c=`[1.0,1.5,2.0]` the model gives almost equal accuracy 0.83.

# ## Apply Stochastic Gradient Descent

# In[35]:


from sklearn.linear_model import SGDClassifier


# In[36]:


clf_sgd=SGDClassifier(alpha=0.001,max_iter=1000,tol=1e-4)
clf_sgd.fit(x_train,y_train)
clf_pred_sgd=clf_sgd.predict(x_test)


# In[37]:


def accuracy(x,y):
    co=0
    for i,j in zip(x,y):
        if i==j:
            co+=1
    accuracy=co/float(len(y))
    return accuracy


# In[38]:


accuracy(clf_pred_sgd,y_test)


# For hyperparameters `alpha=0.001,max_iter=1000` **SGD Classifier** and **LinearSVM** accuracy difference is subtle.

# ## K-fold Cross validation for better choice of <font color='red'>alpha </font> for SGD Classifier

# In[39]:


from sklearn.model_selection import cross_val_score


# In[40]:


clf_sgd_k=[SGDClassifier(alpha=0.00001,max_iter=1000,tol=1e-4),SGDClassifier(alpha=0.00005,max_iter=1000,tol=1e-4),SGDClassifier(alpha=0.0001,max_iter=1000,tol=1e-4),SGDClassifier(alpha=0.0005,max_iter=1000,tol=1e-4)]
alpha=[0.00001,0.00005,0.0001,0.0005]


# In[41]:


for i in range(0,len(alpha)):
    bf=time.time()
    scores=cross_val_score(clf_sgd_k[i],X_mat,rev,cv=10,scoring='accuracy')
    af=time.time()
    print("Alpha:{0:<6f} Accuracy:{1:<20} Time:{2}".format(alpha[i],scores.mean(),(af-bf)))


# **Accuracy** of **SGD Classifier** is greater for **<font color='red'>alpha=0.0005</font>** with **Convergence time=18s**

# In[ ]:




