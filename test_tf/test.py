import pickle
import html
import nltk
from nltk.corpus import stopwords
import re
import string
import pandas as pd
from scipy.sparse import csr_matrix, vstack

# nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def preprocess(text):
    if( type(text) != str):
        text = ''
        return
        
    text = text.lower()
    text = re.sub('\n', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\[.*>\]', '', text)
    text = re.sub('\w*\d\w*', '', text)

    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    text = " ".join(final_text)
    
    return text

VT = pickle.load(open('Vectorize.pkl', 'rb'))
DTC = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))

def check_fake(newsBody):
    newsBody = preprocess(newsBody)
    news = {'Combined': [newsBody]}
    news = pd.DataFrame(data=news, index=[0])
    
    result = DTC.predict( VT.transform(news.Combined))

    return int(result[0])


