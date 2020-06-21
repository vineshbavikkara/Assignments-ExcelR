#### Amazon Review ####
import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs #used to scrap specific content 
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# creating empty reviews list 
AmazonReviews=[]
#extracting reviews from amazon using requests.get()
for i in range(1,20):
    lis=[]
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
    }
    url = 'https://www.amazon.in/Test-Exclusive-606/product-reviews/B07HGJK535/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    response = requests.get(url, timeout=5, headers=HEADERS)
#    print(response.text)
    soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
    reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
    for i in range(len(reviews)):
     lis.append(reviews[i].text)  
    AmazonReviews=AmazonReviews+lis  
    
# writng the reviews to a text file named ReviewAmazon.txt 
with open("ReviewAmazon.txt","w",encoding='utf8') as output:
    output.write(str(AmazonReviews))    
# Joinining all the reviews into single paragraph 
reviewPara = " ".join(AmazonReviews)

# Removing unwanted symbols incase if exists
reviewPara = re.sub("[^A-Za-z" "]+"," ",reviewPara).lower()
reviewPara = re.sub("[0-9" "]+"," ",reviewPara)

# words that contained in one plus mobile phone reviews
reviewWords = reviewPara.split(" ")
print(reviewWords)

#finding and removing the stop words and joining into a single para.
with open("F://stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")
twitter_Words = [w for w in twitter_Words if not w in stopwords]
#joining.
stopPara = " ".join(reviewWords)
#creating a wordcloud
wordcloudStop = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(stopPara)

plt.imshow(wordcloudStop)


# positive words
with open("F:\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
# Positive word cloud
posPara = " ".join ([w for w in twitter_Words if w in poswords])
wordcloudPos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(posPara)
plt.imshow(wordcloudPos)


# negative words  
with open("F:\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
# negative word cloud
negPara = " ".join ([w for w in reviewWords if w in negwords])

wordcloudNeg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(negPara)
plt.imshow(wordcloudNeg)

###############################################################################################################################
###############################################################################################################################
### IMBD Review ###

################# IMDB reviews extraction ######################## Time Taking process as this program is operating the web page while extracting 
############# the data we need to use time library in order sleep and make it to extract for that specific page 
#### We need to install selenium for python
#### pip install selenium
#### time library to sleep the program for few seconds 

from bs4 import BeautifulSoup as bs

#page = "http://www.imdb.com/title/tt0944947/reviews?ref_=tt_urv"
page = "http://www.imdb.com/title/tt6294822/reviews?ref_=tt_urv"

from selenium import webdriver
browser = webdriver.Chrome(executable_path='C:/Users/Lenovo/Downloads/chromedriver')
browser.get(page)

import time
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
reviews = []
i=1
while (i>0):
    #i=i+25
    try:
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
        button.click()
        time.sleep(5)
        ps = browser.page_source
        soup=bs(ps,"html.parser")
        rev = soup.findAll("div",attrs={"class","text"})
        reviews.extend(rev)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break

##### If we want only few recent reviews you can either press cntrl+c to break the operation in middle but the it will store 
##### Whatever data it has extracted so far #######
len(reviews)
len(list(set(reviews)))


import re 
cleaned_reviews= re.sub('[^A-Za-z0-9" "]+', '', reviews)

f = open("reviews.txt","w")
f.write(cleaned_reviews)
f.close()

with open("The_Post.text","w") as fp:
    fp.write(str(reviews))



len(soup.find_all("p"))

###############################################################################################################################
##############################################################################################################################

### Twitter #####

import pandas as pd
import tweepy

#Twitter API credentials
consumer_key = "Kq4mCtnOSPiNwA9ArvYq03DE7"
consumer_secret = "aWBfVbrJWppmEy3mAbrjUHa6Y8AKU6qkCBZwA6ZpAO8BEFaoC2"
access_key = "529590041-eZXHHkluorWkdRZRWiVYW3GVBuvr3VXt84cZcDYA"
access_secret = "rqlG8jzmKTPU3bZoCwgRnOUoD5UYOx8KDjhoXySPrR3mI"

alltweets = []	

def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))                # tweet.get('user', {}).get('location', {})
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    import pandas as pd
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(screen_name+"_tweets.csv")
    return tweets_df

cadd_centre_tweets = get_all_tweets("DreamZoneSchool")

cadd_centre = cadd_centre_tweets['text']


# writng the reviews to a text file named ReviewAmazon.txt 
with open("cadd_centre.txt","w",encoding='utf8') as output:
    output.write(str(cadd_centre))    
# Joinining all the reviews into single paragraph 
twitter_Para = " ".join(cadd_centre)

# Removing unwanted symbols incase if exists
import re
twitter_Para = re.sub("[^A-Za-z" "]+"," ",twitter_Para).lower()
twitter_Para = re.sub("[0-9" "]+"," ",twitter_Para)

# words that contained in one plus mobile phone reviews
twitter_Words = twitter_Para.split(" ")
print(twitter_Words)

#finding and removing the stop words and joining into a single para.
with open("F://stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")
twitter_Words = [w for w in twitter_Words if not w in stopwords]
#joining.
stopPara = " ".join(twitter_Words)
#creating a wordcloud
from wordcloud import WordCloud
import matplotlib.pylab as plt
wordcloudStop = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(stopPara)

plt.imshow(wordcloudStop)


# positive words
with open("F:\\positive_words.txt","r") as pos:
  poswords = pos.read().split("\n")
# Positive word cloud
posPara = " ".join ([w for w in twitter_Words if w in poswords])
wordcloudPos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(posPara)
plt.imshow(wordcloudPos)


# negative words  
with open("F:\\negative_words.txt","r") as neg:
  negwords = neg.read().split("\n")
# negative word cloud
negPara = " ".join ([w for w in twitter_Words if w in negwords])

wordcloudNeg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(negPara)
plt.imshow(wordcloudNeg)


## Twitter sentiment analysis ; polarity and subjectivity

# Importing libraries
import tweepy
from textblob import TextBlob as tb

#Twitter API credentials
consumer_key = "Kq4mCtnOSPiNwA9ArvYq03DE7"
consumer_secret = "aWBfVbrJWppmEy3mAbrjUHa6Y8AKU6qkCBZwA6ZpAO8BEFaoC2"
access_key = "529590041-eZXHHkluorWkdRZRWiVYW3GVBuvr3VXt84cZcDYA"
access_secret = "rqlG8jzmKTPU3bZoCwgRnOUoD5UYOx8KDjhoXySPrR3mI"

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key,access_secret)
api = tweepy.API(auth)
public_tweets = api.search('DreamZoneSchool')
for tweet in public_tweets:
    print(tweet.text)
    analysis = tb(tweet.text)
    print(analysis.sentiment)

#################################################################################################################################