

# Importing Essential Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords

import pickle


# In[1]:


# Reading from CSV file into DataFrame

df = pd.read_csv('/content/drive/MyDrive/fake-news/train.csv')
df.shape


# In[ ]:


# Filtering Null Attributes

df.dropna(inplace=True)
df.shape


# ## Sampling

# In[ ]:


sample_size = 10000
df = df.sample(sample_size)


# In[ ]:


fil = open("stats.txt", "a")
fil.write("\nSample Size: "+ str(sample_size))
scores = {"Sample_Size": sample_size}


# ## Data Preprocessing

# In[ ]:


nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def preprocess(text):
    if( type(text) != str):
        text = ''
        return

    # Coverting text to lowercase

    text = text.lower()

    # Removing Newline Characters

    text = re.sub('\n', '', text)

    # Removing Whitespaces

    text = re.sub('\\W', ' ', text)

    # Removing Square Brackets

    text = re.sub('\[[^]]*\]', '', text)

    # Removing URL's

    re.sub(r'http\S+', '', text)

    # Other Preprocessing tasks
    text = re.sub('\[.*>\]', '', text)
    text = re.sub('\w*\d\w*', '', text)

    # Removing Stop Words

    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    text = " ".join(final_text)

    # Returning the processed text
    
    return text


# In[ ]:


# Combining all attributes into single attribute

df['combined'] = df['author'] + ' ' + df['title'] + ' ' + df['text']

# Applying preprocessing tasks to combined attributes

df['combined'] = df['combined'].apply(preprocess)


# ## Data Analysis

# In[ ]:


# Calculating Title Length, Text Length

df['title_length'] = [len(x) for x in df['title']]
df['text_length'] = [len(x) for x in df['text']]


# Calculating No. of words in Title and Text

df['title_words'] = [len(x.split()) for x in df['title']]
df['text_words'] = [len(x.split()) for x in df['text']]

df.head(2)


# Comparing Between True and Fake Articles

# In[ ]:


true_news = df.loc[df['label'] == 0]
fake_news = df.loc[df['label'] == 1]

plt.bar(['Fake', 'Not Fake'], [fake_news.size, true_news.size], color=['m','g'])
plt.xlabel('News Category')
plt.ylabel('Count')
plt.title('Distribution of Fake and Not Fake News')

plt.savefig('/content/drive/MyDrive/fake_news_img/dist_comp_'+str(sample_size)+'.png', dpi=300)


# In[ ]:


from matplotlib.pyplot import figure


# In[ ]:


figure(figsize = (20,15))

plt.subplot(421)
buckets_title = np.arange(0, 210, 10)
plt.hist(true_news.title_length, rwidth=0.5, bins=buckets_title)
plt.title('Title Length of True News Articles')
plt.ylabel('Title Length')

plt.subplot(422)
plt.hist(fake_news.title_length, rwidth=0.5, bins=buckets_title)
plt.title('Title Length of Fake News Articles')
plt.ylabel('Title Length')

plt.subplot(423)
buckets_text = np.arange(0, 22000, 2000)
plt.hist(true_news.text_length, rwidth=0.5, bins=buckets_text)
plt.title('Text Length of True News Articles')
plt.ylabel('Text Length')

plt.subplot(424)
plt.hist(fake_news.text_length, rwidth=0.5, bins=buckets_text)
plt.title('Text Length of Fake News Articles')
plt.ylabel('Text Length')

plt.subplot(425)
title_words_bucket = np.arange(0, 60, 10)
plt.hist(true_news.title_words, rwidth=0.5, bins=title_words_bucket)
plt.title('Title Word Count of True News Articles')
plt.ylabel('Title Word Count')

plt.subplot(426)
plt.hist(fake_news.title_words, rwidth=0.5, bins=title_words_bucket)
plt.title('Title Word Count of Fake News Articles')
plt.ylabel('Title Word Count')

plt.subplot(427)
text_words_bucket = np.arange(0, 5500, 500)
plt.hist(true_news.text_words, rwidth=0.5, bins=text_words_bucket)
plt.title('Text Word Count of True News Articles')
plt.ylabel('Text Word Count')

plt.subplot(428)
plt.hist(fake_news.text_words, rwidth=0.5, bins=text_words_bucket)
plt.title('Text Word Count of Fake News Articles')
plt.ylabel('Text Word Count')

plt.savefig('/content/drive/MyDrive/fake_news_img/comparision_'+str(sample_size)+'.png', dpi=300)


# ### True News Articles Statistics

# In[ ]:


true_news.drop(['id', 'label'], axis=1).describe().loc[['mean','std']].astype(int)


# ### Fake News Articles Statistics

# In[ ]:


fake_news.drop(['id', 'label'], axis=1).describe().loc[['mean', 'std']].astype(int)


# ## Model Training

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df['combined'], df['label'], test_size=0.2, random_state=2)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorization = TfidfVectorizer(stop_words="english")
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# ### Logistic Regression

# In[ ]:


LR = LogisticRegression()
LR.fit(xv_train, y_train)
y_pred = LR.predict(xv_test)


# In[ ]:


scores['LR'] = [LR.score(xv_test, y_test)]


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# ### Decision Tree Classification

# In[ ]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
y_pred = DT.predict(xv_test)


# In[ ]:


scores['DT'] = [DT.score(xv_test, y_test)]


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# ### Gradient Boosting Classifier

# In[ ]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
y_pred = GBC.predict(xv_test)


# In[ ]:


scores['GBC'] = [GBC.score(xv_test, y_test)]


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# ### Random Forest Classifier

# In[ ]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
y_pred = RFC.predict(xv_test)


# In[ ]:


scores['RFC'] = [RFC.score(xv_test, y_test)]


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# ### Passive Aggressive Classifier

# In[ ]:


PAC = PassiveAggressiveClassifier(max_iter=50)
PAC.fit(xv_train, y_train)
y_pred = PAC.predict(xv_test)

scores['PAC'] = [PAC.score(xv_test, y_test)]

print(confusion_matrix(y_test, y_pred))


# ### Naive Bayes Classifier

# In[ ]:


NBC = MultinomialNB()
NBC.fit(xv_train, y_train)
y_pred = NBC.predict(xv_test)

scores['NBC'] = [NBC.score(xv_test, y_test)]

print(confusion_matrix(y_test, y_pred))


# ### Support Vector Machine

# In[ ]:


SV = SVC()
SV.fit(xv_train, y_train)
y_pred = SV.predict(xv_test)

scores['SVC'] = [SV.score(xv_test, y_test)]

print(confusion_matrix(y_test, y_pred))


# In[ ]:


scores


# Writing Scores to Files

# In[ ]:


from pathlib import Path

path = Path('/content/drive/MyDrive/stats.csv')
if not path.is_file():
  # stat = pd.DataFrame(columns=['Sample_Size', 'LR', 'DTC', 'GBC', 'RFC'])
  stat = pd.DataFrame(scores)
else:
 b

if sample_size not in stat.Sample_Size.unique():
  stat = pd.concat([stat, pd.DataFrame(scores)])

stat.sort_values(by='Sample_Size', inplace=True)

stat.to_csv('/content/drive/MyDrive/stats.csv', mode='w', index=False)
stat.to_csv('stats.csv', mode='w', index=False)


# In[45]:


stat = pd.read_csv('/content/drive/MyDrive/stats.csv')


# In[ ]:


fil.write(str(scores))
fil.close()


# In[ ]:


# stat = stat[3:]


# In[47]:


stat.Sample_Size = stat.Sample_Size.astype(str)


# In[48]:


figure(figsize=(16,9))
plt.plot(stat.Sample_Size, stat.LR, label="Logistic Regression")
plt.plot(stat.Sample_Size, stat.DT, label="Decision Tree")
plt.plot(stat.Sample_Size, stat.GBC, label="Gradient Booster")
plt.plot(stat.Sample_Size, stat.RFC, label="Random Forest Classifier")
plt.plot(stat.Sample_Size, stat.PAC, label="Passive Aggressive Classifier")
plt.plot(stat.Sample_Size, stat.NBC, label="Naive Bayes Classifier")
plt.plot(stat.Sample_Size, stat.SVC, label="Support Vector Classifier")
plt.xlabel('Sample Size')
plt.ylabel('Accuracy Score')
plt.legend()

plt.savefig('/content/drive/MyDrive/fake_news_img/graph_'+str(sample_size)+'.png', dpi=300)


# In[ ]:


scores


# In[ ]:


stat.drop(labels=[0,1,2,3], inplace=True)


# In[ ]:


stat


# In[ ]:


dp = stat.loc[stat.Sample_Size == str(sample_size)]
dp

for key in list(scores.keys()):
  stat.loc[stat.Sample_Size == str(sample_size)][key] = scores[key]

# list(scores.keys())


# In[ ]:




