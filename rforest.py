#Import neccessary libraries
import numpy as np
import pandas as pd
import json, nltk, re
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB # Naive Bayes Classifier
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import preprocessor as p
from sklearn.ensemble import RandomForestClassifier


#Read our dataset corpus
corpus = pd.read_csv("dataset.csv")
#Set global display options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 690)

#Column name as variables
#Corpus is of following csv format: TweetID, Tweet, User, Label)
corpus = corpus.drop('TweetID',1)

tweet = corpus.columns.values[0]
sentiment = corpus.columns.values[2]


# function to clean the dataset (combining library tweet-preprocessor and simple regex)
def clean_tweets(df):
  tempArr = []
  #set up punctuations we want to be replaced
  withoutspace = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
  withspace = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

  for line in df:
    # call clean() from tweet_processor, processing
    tweetInProcess = p.clean(line)
    # Remove puctuation from tweet (converted to lowercase)
    tweetInProcess = withoutspace.sub("", tweetInProcess.lower()) 
    tweetInProcess = withspace.sub(" ", tweetInProcess)

    #Finally add to our list
    tempArr.append(tweetInProcess)
  return tempArr

#corpus['ProcessedTweet'] = np.vectorize(process_tweet)(corpus[tweet])

# Cleaning of dataset
train_tweet = clean_tweets(corpus["Text"])
train_tweet = pd.DataFrame(train_tweet)
# append cleaned tweets to the training data
corpus["ProcessedTweet"] = train_tweet

#Removing Stop Words
#Stop words are the words that do not add any meaning to the sentence in terms of Natural Language Processing (NLP).
#Eg: “I”, “me”, “my”, “myself”, “we”, “our”, “ours”, etc.
stop = stopwords.words('english')
corpus['ProcessedTweet'] = corpus['ProcessedTweet'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in stop))

#Split data into training and testing datasets 
x_train, x_test, y_train, y_test =  train_test_split(corpus["ProcessedTweet"], corpus["Label"], 
                                                      test_size = 0.2, random_state = 10)

count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(sublinear_tf=True)
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)


model = RandomForestClassifier(n_estimators = 400)
model.fit(x_train_tfidf,y_train)
predictions = model.predict(x_test_tfidf)

print("Accuracy score for Random Forrest Classification is: ", 
  accuracy_score(y_test, predictions)*100, '%')
print(classification_report(y_test, predictions))


#Confusion Matrix 
from sklearn.metrics import confusion_matrix,f1_score
a = confusion_matrix(y_test,predictions)
plt.figure(dpi=150)
sns.heatmap(a.T, annot = True, fmt='d', cbar=False)

plt.title('Confusion Matrix for Random Forrest')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()

#References:
#https://github.com/importdata/Twitter-Sentiment-Analysis/blob/master/Twitter_Sentiment_Analysis_Support_Vector_Classifier.ipynb
