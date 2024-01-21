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

#Read our dataset corpus
corpus = pd.read_csv("dataset.csv")
#Set global display options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 690)

#Column name as variables
#Corpus is of following csv format: TweetID, Tweet, User, Label)
corpus = corpus.drop('TweetID',1)
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

#Lemmatization
#nltk.download('omw-1.4')
#lemmatizer = WordNetLemmatizer()
#corpus['ProcessedTweet']  = corpus['ProcessedTweet'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

''' Stemming
# stemmer = PorterStemmer()

# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
'''
'''
for i in range(len(tokenized_tweet)):
    
     # Below code is used for no stop word removal
#     tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) 
     tokenized_tweet[i] = ' '.join([word for word in tokenized_tweet[i] if word not in stop_words])  

corpus['ProcessedTweet'] = tokenized_tweet
corpus['TextBlobScore'] = corpus['ProcessedTweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
corpus['Sentiment'] = corpus['TextBlobScore'].apply(lambda c: 'Positive' if c == 1 else 'Negative')


fig = plt.figure(figsize=(8, 5))
corpus['TextBlobScore'].hist()
plt.xlabel('Polarity Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.title("TextBlob Sentiment Analysis")'''
#plt.show()

count_vectorizer = CountVectorizer(ngram_range=(1,2))    # Unigram and Bigram
vectorizedTweets = count_vectorizer.fit_transform(corpus['ProcessedTweet'])  

#tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
#vectorizedTweets = tf_idf_vectorizer.fit_transform(corpus['ProcessedTweet'])

X_train, X_test, y_train, y_test = train_test_split(vectorizedTweets, corpus[sentiment],
                                                    test_size = 0.2, random_state = 99)

#MODEL 1: Multinomial Naive Bayes

modelMNB = MultinomialNB().fit(X_train, y_train) 
predicted_naive = modelMNB.predict(X_test)
scoreMNB = accuracy_score(predicted_naive, y_test)
print("Accuracy with Multinomial Naive Bayes classification: ",scoreMNB)
print('\n')
print(classification_report(y_test, predicted_naive))

plt.figure(dpi=100)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix for Multinomial Naive Bayes')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.legend()
plt.show()

#MODEL 2: Bernoulli Naive Bayes
BNBmodel = BernoulliNB().fit(X_train, y_train)
predicted_bernnb = BNBmodel.predict(X_test)
score_bernnb = accuracy_score(predicted_bernnb, y_test)
print("Accuracy for Bernoulli Naive Bayes classification: ", score_bernnb)
print(classification_report(y_test, predicted_bernnb))

#MODEL 3: Logistic Regression 
LRmodel = LogisticRegression(C = 2, max_iter = 100, n_jobs = -1).fit(X_train, y_train)
predicted_lr = LRmodel.predict(X_test)
score_lr = accuracy_score(predicted_lr, y_test)
print("Accuracy for Logistic Regression classifier: ", score_lr)
print(classification_report(y_test, predicted_lr))

plt.figure(dpi=100)
mat1 = confusion_matrix(y_test, predicted_lr)
sns.heatmap(mat1.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix for LR')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# calculate the fpr and tpr for all thresholds of the classification
probs = modelMNB.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(dpi=120)                       
plt.title('ROC Curve: Multinomial Naive Bayes')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate', color='g')
plt.xlabel('False Positive Rate', color='r')
plt.show()

#References:
#https://github.com/Gunjan933/twitter-sentiment-analysis/blob/master/twitter-sentiment-analysis.ipynb
#https://stackoverflow.com/questions/63378920/attributeerror-module-preprocessor-has-no-attribute-clean
#https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/
#https://stackoverflow.com/questions/16729574/how-to-get-a-value-from-a-cell-of-a-dataframe
#https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/
