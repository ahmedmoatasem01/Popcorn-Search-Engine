import os
import pandas as pd
import pyterrier as pt
import requests
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words("english")
ps = PorterStemmer()

def removeStopwords(text):
  tokens = word_tokenize(text)
  filtered_tokens = [word.lower() for word in tokens if word not in stop_words]
  return " ".join(filtered_tokens)

def steem(text):
  tokens = word_tokenize(text)
  stemmed_tokens = [ps.stem(word) for word in tokens]
  return " ".join(stemmed_tokens)

def clean(text):
   text = str(text).lower()
   text = re.sub(r'[^a-z\s]', ' ', text)
   text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)
   text = re.sub(r"\s+", " ", text).strip()
   return text

def processText(text, stem=True):
      text = clean(text)
      text = removeStopwords(text)
      if stem:
          text = steem(text)
      return text



