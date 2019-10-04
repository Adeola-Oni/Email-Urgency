import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import re   
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
        
data = pd.read_csv('Data2.csv', encoding = 'ISO-8859-1')
data.head()

def freq_words(text):
    word_list = []
    
    for x in text.split(' '):
        word_list.extend(x)
        
    word_freq = pd.Series(word_list).value_counts()
    word_freq[:20]
    return word_freq

corpus = []
for i in range(45):
    cleaned = re.sub('[^a-zA-Z]', ' ',  data['Email Body'].values[i])
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    ps = PorterStemmer()
    cleaned = [ps.stem(word) for word in cleaned if not word in set(stopwords.words('english'))]
    cleaned = ' '.join(cleaned)
    corpus.append(cleaned)
    
from wordcloud import WordCloud
#Generate word cloud
cor = pd.Series( (v for v in corpus) )
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(freq_words(cor.str))
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

countvectorizer = CountVectorizer()
countvectorizer.fit(corpus)

print("content", countvectorizer.vocabulary_)

X = countvectorizer.transform(corpus).toarray()

y = data['Urgency'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
cm = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(cm) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred, normalize=False) )
print ('Report : ')
print (classification_report(y_test, y_pred) )

names = np.unique(y_pred)
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


newValue = input('Please input the body of the email: ')

newcorpus = []
newCleaned = re.sub('[^a-zA-Z]', ' ',  newValue)
newCleaned = newCleaned.lower()
newCleaned = newCleaned.split()
newCleaned = [ps.stem(word) for word in newCleaned if not word in set(stopwords.words('english'))]
newCleaned = ' '.join(newCleaned)
newcorpus.append(newCleaned)
newX = countvectorizer.transform(newcorpus).toarray()
newPrediction = nb.predict(newX)

print(newPrediction)

