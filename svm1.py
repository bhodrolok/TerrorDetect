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
from sklearn import svm
import seaborn as sns
import preprocessor as p

#Read our dataset corpus
corpus = pd.read_csv("dataset.csv")
#Set global display options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 690)

#Column name as variables
#Corpus is of following csv format: TweetID, Tweet, User, Label)
corpus = corpus.drop('TweetID')
#TweetID is unneccesarry

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

# Dataset cleaning, pass in the Text i.e. raw tweets column
train_tweet = clean_tweets(corpus["Text"])
train_tweet = pd.DataFrame(train_tweet)
# append cleaned tweets to the training data
corpus["ProcessedTweet"] = train_tweet

#Removing Stop Words
#Stop words are the words that do not add any meaning to the sentence in terms of Natural Language Processing (NLP).
#Eg: “I”, “me”, “my”, “myself”, “we”, “our”, “ours”, etc.
stop = stopwords.words('english')
corpus['ProcessedTweet'] = corpus['ProcessedTweet'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in stop))

# extract the labels from the train data
y = corpus.Label.values

# use 80% for the training and 20% for the test data sets
x_train, x_test, y_train, y_test = train_test_split(corpus.ProcessedTweet.values, y, 
                                                    stratify = y, 
                                                    random_state = 1, 
                                                    test_size = 0.2, shuffle = True)

# vectorize tweets for building model
vectorizer = CountVectorizer(binary = True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw tweets
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# classify using support vector classifier
svc = svm.SVC(kernel = 'linear', probability = True)

# fit the SVC model based on the given training data
prob = svc.fit(x_train_vec, y_train).predict_proba(x_test_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svc.predict(x_test_vec)

print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')

print(classification_report(y_test, y_pred_svm))

a = confusion_matrix(y_test, y_pred_svm)
plt.figure(dpi=420)
sns.heatmap(a.T, annot=True, fmt='d', cbar = False)

plt.title('Confusion Matrix for SVM')
plt.xlabel('True label')
plt.ylabel('Predicted label')
#plt.savefig("assets/confusion_matrix.png")
plt.legend()
plt.show()