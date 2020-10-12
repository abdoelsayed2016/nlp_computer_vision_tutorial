
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re


stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()
counter=0


def html_to_tet(review):
    review_text=BeautifulSoup(review,"lxml").get_text()
    if len(review_text)==0:
        review_text=review
    review_text=re.sub(r"\<.*\>","",review_text)
    try:
        review_text=review_text.encode('ascii',"ignore").decode('ascii')
    except UnicodeDecodeError:
        review_text=review_text.decode("ascii","ignore")
    return review_text

def letters_only(text):
    return re.sub("[^a-zA-Z]"," ",text)

def clean_review(review):
    return letters_only(html_to_tet(review)).lower()

def lemmatize(tokens: list)-> list:
    tokens=list(map(lemmatizer.lemmatize,tokens))
    lemmatized_tokens=list(map(lambda x:lemmatizer.lemmatize(x,"v"),tokens))
    meaningful_words=list(filter(lambda x : not x in stop_words,lemmatized_tokens))
    return meaningful_words
def preprocess(review: str , total:int,show_progress:bool=True) ->list:
    
    if show_progress:
        global counter
        counter+=1
        print('Processing .... %6i/%6i'%(counter,total),end='\r')
    review=clean_review(review)
    tokens=word_tokenize(review)
    lemmas=lemmatize(tokens)
    return lemmas