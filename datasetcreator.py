import snscrape.modules.twitter as sntwitter
import pandas as pd

# Scraping data and append tweets to list
terrorKeywordlist = ['ISIS','Jihad','Kafir','Amaq Agency','Crusader Army','Jihad for Ummah','Jihad in the Quran','How to do Jihad','Killing Infidels','Soldiers of the Caliphate']
for eachKeyword in terrorKeywordlist:
	tweets_list = []
	queryBase = ' since:2021-01-01 until:2021-08-29 lang:en'
	for i, tweet in enumerate(sntwitter.TwitterSearchScraper(eachKeyword + queryBase).get_items()):
		if i > 10:
			break
		tweets_list.append([tweet.id, tweet.content, tweet.user.username])

	tempdf = pd.DataFrame(tweets_list, columns = ['TweetID','Text','Username'])
	filenametemp = eachKeyword + "1.csv"
	tempdf.to_csv(filenametemp, index=False)

print("Operation complete.")

#References:
#https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/
#https://github.com/JustAnotherArchivist/snscrape