---
layout: post
title: Rule-Based Sentiment Analysis on Twitter Data
---
![Alternate image text](/images/twitter/social_media_sen.jpg)

# Part 2: Sentiment Analysis 

## Why in the world would someone care about data from social media?
With an abundance of social media, news articles, and opinion pieces available on the web, Data Scientists are able to utilize these resources in order to extract millions of text data. In this project, I am specifically looking at tweets from the 2020 United States Presidential Election. To read how I collected these tweets, please check out my previous [post!](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"} 

Tweet data is advantageous because people use twitter to express opinions and engage with others publicly. From twitter, Data Scientists are able to find key words amongst tweets, analyze geographic differences in opinions about topics, detect bots on twitter, analyze user engagement for specific topics, and create a timeline of events from tweet data (Chen et al., 2020). As well, tweet data can even be utilized to make predictions such as the [2020 US Presidential Elections](https://www.independent.co.uk/news/world/americas/us-election-2020/2020-election-whos-going-to-win-ai-trump-biden-results-outcome-odds-b1374290.html){:target="_blank"}! Pretty amazing, right? 

We know that social media influences opinions. I have personally experienced this myself by following people who tend to have a certain viewpoint who then shares resources, links, and more education on that view. In result, my opinion grows stronger towards that topic. The spread and influence of opinions on social media can positively and negatively impact society. An example of negative impact is the rise of hate groups and terrorist organizations through the help of social media. [According to Christopher Way, Director of FBI](https://www.fbi.gov/news/testimony/worldwide-threats-to-the-homeland-091720){:target="_blank"}, these dangerous organizations utilize social media platforms to recruit, radicalize vulnerable persons in the U.S., propagate its ideology, and create false personas on social media to discredit U.S. individuals and institutions. Thus, analyzing and understanding the intricacies of social media can help pinpoint how, where, and why radical groups spread on social media in order to mitigate the problem. 

On the business spectrum, one might want to utilize tweet data to identify influencers, key markets, and trends in that market in order to implement successful marketing campaigns. 

Alright, at this point you are probably thinking to yourself, "Wow! Collecting and analyzing data from social media is super cool!"

## Purpose
For my project, I am focusing on the context of popular topics from collecting tweet data regarding the 2020 US Presidential election. The goal of this is to apply sentiment analysis and topic modelling to these tweets. **Sentiment Analysis** specifically is a Natural Language Process in order to detect and analyze opinions or attitude within tweets. 

**A few questions to answer with sentiment analysis and topic modelling:**

- How do the majority of tweets collected perceive Trump and Biden in total versus for each state? Does the perception change from June 2020 to October 2020?
- What is the sentiment on specific topics that were important to the election such as COVID19, voter fraud, BLM, supremacist groups, Trump tax return, Supreme Court, anti-mask.
- How do tweets posted from a twitter account user with a large following perceive certain topics?
- What are the top 5 positively perceived topics versus top 5 negatively perceived topics?

## Methods

#### Rule Based
Rule-based sentiment analysis calculates a sentiment score on a text based off implemented rules such as determining if negations are present or a specific word is present. 

- TextBlob
- Vader Sentiment Analysis

#### Machine Learning
Machine learning can identify the sentiment if you are able to train a dataset with an independent variable. For example, machine learning can be utilized to predict the score from Yelp reviews, Airbnb reviews, and movie reviews. You could say that the sentiment of 5 out of 5 stars is very positive and 1 out of 5 stars is very negative. 

If you want to use machine learning to identify sentiment but you do not have a rating. There is a work around method where you can label a training dataset in order to create an independent variable for sentiment. The problem with this is that because we are able to collect millions of tweets. Identifying the sentiment for even 1000 tweets by hand is not enough data to train a model. 

- SVM
- Naive Bayes Classifier 

## Text Blob

[Text Blob](https://textblob.readthedocs.io/en/dev/){:target="_blank"} is a text processing package that works with part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

**Features**
- Noun phrase extraction
- Part-of-speech tagging
- Sentiment analysis
- Classification (Naive Bayes, Decision Tree)
- Tokenization (splitting text into words and sentences)
- Word and phrase frequencies
- Parsing
- n-grams
- Word inflection (pluralization and singularization) and lemmatization
- Spelling correction
- Add new models or languages through extensions
- WordNet integration

From TextBlob, the sentiment returns a polarity and subjectivity score. The polarity score output is a float between the range [-1.0, 1.0], where -1.0 is 100% negative and 1.0 is 100% positive. The subjectivity is a float within the range [0.0, 1.0] where 1.0 is very subjective (influence by a personal feeling or opinion) and 0.0 is very objective (not influenced by personal feelings or opinion).

#### Step 1: Import libraries
``` python
from textblob import TextBlob
```

#### Step 2: Clean text data 

Let’s take a look at a tweet example from the dataset: 
      
![Alternate image text](/images/twitter/original_tweet.png)

The first thing to note from this tweet is that the subjectivity from the viewpoint of the twitter account user is negative.

Things to note about the text:
1.  Punctuation: ‘@’,’!’, ‘…’, ‘:’, ‘/’, ‘.’, ‘?’
2.  Capitalization of words
3.  Face Palm emoji which displays the  emotion frustration or  disappointment
4. @mention of twitter user realDonaldTrump
5. Negation word: Can’t 
In order to improve the accuracy when processing the tweet data with TextBlob, I first clean the text data by changing uppercase letters to lowercase, remove punctuation and emojis, remove consecutive spaces. Other important cleaning capabilities to account for within social media text is removing hyperlinks, removing newlines, and removing the retweet account when tweets are retweeted.

#### Use Regex to clean the data:

``` python

def cleanTxt(text):
        
    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the account retweeted from
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub('https', '', text)
    text = re.sub('@', '', text)
    text = re.sub(r'[^\w\s]', '', text) # removes punctuation and emojis
    text = re.sub(r'\s+', ' ', text)  # replace consecutive spaces
    text = re.sub(r'\s*<br\s*/?>\s*', '\n', text)  # newline after a <br>
    text = re.sub(r'^\s+', '', text)  # remove spaces at the beginning
    text = re.sub(r'\s+$', '', text)  # remove spaces at the end

    return text

# Clean text
Text = df['full_text'].apply(cleanTxt)

# Add lower cases
for i in np.arange(0, len(Text), 1):
    Text[i] = Text[i].lower()

```
<br>
After cleaning the text, the tweet now looks like this:<br><br>
![Alternate image text](/images/twitter/clean_text.png)
<br>

#### Step 3: Create a function to get the subjectivity and polarity
``` python
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return  TextBlob(text).sentiment.polarity
```

#### Step 4: Create two new columns 'Subjectivity' & 'Polarity'

``` python
df['Subjectivity'] = df['Full_Text'].apply(getSubjectivity)
df['Polarity'] = df['Full_Text'].apply(getPolarity)
```

#### Step 5: Create a function to define sentiment by calculating negative as (Polarity < 0), neutral as (Polarity = 0), and positive as (Polarity > 0) analysis
``` python

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['Polarity'].apply(getAnalysis)
```
#### Step 6: Compare the sentiment score before cleaning the data with the score after cleaning the data:

TextBlob Analysis without cleaning the text<br>
![Alternate image text](/images/twitter/sentiment_unclean.png)
<br><br>
TextBlob Analysis with cleaning the text<br>
![Alternate image text](/images/twitter/sentiment_clean.png)
<br>
The subjectivity and polarity for this specific text did not change after cleaning the text. This somewhat makes sense because the text is already fairly simplified because there are no links present, no additional lines or spaces. 

In order to compare the entire text dataset that is original content (not including retweets) for the month of  August, the 10% of the text dataset calculated polarity changed after cleaning  the dataset.<br>
![Alternate image text](/images/twitter/bar_unclean_clean.png)




While there is not a significant change in polarity after cleaning the text, it is still important  to remove anything that  might hinder the  analysis  for example html links, extra space and lines, punctuation, and emojis.


#### Step 7: Quick Analysis on comparing sentiment in relation to the 2020 Presidential Election within August
![Alternate image text](/images/twitter/August_Sentiment.png)

### To Be Continued …
