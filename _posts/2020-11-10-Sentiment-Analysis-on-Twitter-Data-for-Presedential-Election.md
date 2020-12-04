---
layout: post
title: Rule-Based Sentiment Analysis on Twitter Data
---
![Alternate image text](/images/twitter/social_media_sen.jpg)

# Part 2: Sentiment Analysis 

## Why in the world would someone care about social media data?
With an abundance of social media, news articles, and opinion pieces available on the web, Data Scientists are able to utilize these resources in order to extract millions of text data. In this project, I am specifically looking at tweets from the 2020 United States Presidential Election. To read how I collected these tweets, please check out my previous [post!](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"} 

Extracting data from twitter is greatly beneficial because twitter is used to express opinions and engage with others publicly. From twitter, Data Scientists are able to find key words amongst tweets, analyze geographic differences in opinions about topics, detect bots on twitter, analyze user engagement for specific topics, and create a timeline of events from tweet data (Chen et al., 2020). As well, tweet data can even be utilized to make predictions such as the [2020 US Presidential Elections](https://www.independent.co.uk/news/world/americas/us-election-2020/2020-election-whos-going-to-win-ai-trump-biden-results-outcome-odds-b1374290.html){:target="_blank"}! Pretty amazing, right? 

We as well know that social media influences opinions. It starts with following those who tend to have a certain viewpoint who then shares resources, links, and more education towards that point of view. In result, our opinions grow stronger towards that topic. The spread and influence of opinions on social media can positively and negatively impact society. An example of negative impact is the rise of hate groups and terrorist organizations through the help of social media. [According to Christopher Way, Director of FBI](https://www.fbi.gov/news/testimony/worldwide-threats-to-the-homeland-091720){:target="_blank"}, these dangerous organizations utilize social media platforms to recruit, radicalize vulnerable persons in the U.S., propagate its ideology, and create false personas on social media to discredit U.S. individuals and institutions. 

Therefore, we can see how analyzing and understanding the intricacies of social media can help pinpoint how, where, and why trending opinions spread. This might interest businesses as well because a compnay may want to utilize influencers who will positive influence their business, and identify key trends within that market in order to implement successful marketing campaigns on social media. 

Alright, at this point you are probably thinking to yourself, "Wow! Data from social media is super cool!"

## Purpose
For my project, I am focusing on the context of popular topics from collecting tweet data regarding the 2020 US Presidential election. The goal of this is to apply sentiment analysis and topic modelling to these tweets. **Sentiment Analysis** specifically is a Natural Language Process in order to detect and analyze opinions or attitudes within text. 

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
Machine learning can identify the sentiment if you are able to train a dataset with an independent variable. For example, machine learning can be utilized to predict the score from Yelp reviews, Airbnb reviews, and movie reviews. You could label the sentiment of 5 out of 5 stars very positive and 1 out of 5 stars very negative. 

If we want to use machine learning to identify sentiment but do not have a rating. There is a work around method by labelling a training dataset in order to create an independent variable for sentiment. The problem with this is that if we collect hundres of thousands and millions of data, labeling the sentiment for even 1000 tweets by hand is not enough data to train a model. 

- SVM
- Naive Bayes Classifier 

## TEXT BLOB

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

From TextBlob, the sentiment returns a polarity and subjectivity score. 

**Polarity** score output is a float between the range [-1.0, 1.0], where -1.0 is 100% negative and 1.0 is 100% positive. 

**Subjectivity** is a float within the range [0.0, 1.0] where 1.0 is very subjective (influence by a personal feeling or opinion) and 0.0 is very objective (not influenced by personal feelings or opinion).

#### Step 1: Import libraries
``` python
from textblob import TextBlob
```

#### Step 2: Clean text data 

Let’s take a look at a tweet example from the dataset: 
      
![Alternate image text](/images/twitter/original_tweet.png)

Things to note about the text:
1.  Punctuation: ‘@’,’!’, ‘…’, ‘:’, ‘/’, ‘.’, ‘?’
2.  Capitalization
3.  Face Palm emoji which signifies frustration or  disappointment
4. @mention of twitter user realDonaldTrump

In order to improve the accuracy when processing the tweet data with TextBlob, I first clean the text data by changing uppercase letters to lowercase, remove punctuation and emojis, remove consecutive spaces, remove hyperlinks, removing newlines, and remove the retweet account when tweets are retweeted.

#### Use Regex to clean the data:

``` python

def cleanTxt(text):
        
    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the account retweeted from
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub('https', '', text)  #remove word https
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
#### Step 6: Compare sentiment score before cleaning the data with after cleaning the data:

In order to compare the entire text dataset that is original content for the month of August, the 10% of the text dataset calculated polarity changed after cleaning the dataset.<br>
![Alternate image text](/images/twitter/textblob_unclean_table.png)

![Alternate image text](/images/twitter/textblob_unclean_clean.png)

It is important to note that while the polarity did change for 10% of the text after cleaning the dataset, the polarity did not change enough to change the sentiment. The reason for cleaning text even if it may seem not significant is because html links, extra space and lines, punctuation, and emojis may hinder the analysis.


#### Step 7: Quick Analysis on comparing sentiment in relation to the 2020 Presidential Election within August
![Alternate image text](/images/twitter/august_textblob_sentiment.png)

## VADER SENTIMENT ANLAYSIS

Social media text is complex because there are emojis to express feelings, acronyms (LOL OMG LMAO ROFL WTF ASAP), intentionally misspelled words like sucks -> sux and fav -> favorite. There are as well slang words that are used on social media that is not identified in the dictionary at least yet such as yolo, muah, haha, woohoo, and using punctuation to make an emotion or face such as (: (; <3 . We as well utilize words in different context. For example, on social media, one might use the word wicked to mean cool/awesome which is then taken an originally negative word and utilizing it in a positive way. These complexities in social media need to be accounted for when identifying sentiment in a sentence. Luckily for us, a package called [Vader Sentiment Analysis](https://github.com/cjhutto/vaderSentiment){:target="_blank"} has done this for us! 

Vader Sentiment Analysis is another lexicon rule-based sentiment analysis tool that was specifically developed for social media text. Vader Sentiment accounts for speed and performance which is important for large datasets such as thousands or millions of tweets.

Score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).

Examples of typical use cases for sentiment analysis, including proper handling of sentences with:
- typical negations (e.g., "not good")
- use of contractions as negations (e.g., "wasn't very good")
- conventional use of punctuation to signal increased sentiment intensity (e.g., "Good!!!")
- conventional use of word-shape to signal emphasis (e.g., using ALL CAPS for words/phrases)
- using degree modifiers to alter sentiment intensity (e.g., intensity boosters such as "very" and intensity dampeners such as "kind of")
- understanding many sentiment-laden slang words (e.g., 'sux')
- understanding many sentiment-laden slang words as modifiers such as 'uber' or 'friggin' or 'kinda'
- understanding many sentiment-laden emoticons such as :) and :D
- translating utf-8 encoded emojis such as 💘 and 💋 and 😁
- understanding sentiment-laden initialisms and acronyms (for example: 'lol')

### Clean Text
This time when cleaning the text, for vader sentiment analysis. I do not make

``` python
def cleanTxt(text):

        #initializing punctuations string  
    punc = '''()-[]{};:'"\, <>./?@#$%^&*_~'''
  
    # Removing punctuations in string 
    # Using loop + punctuation string 
    for ele in text:  
        if ele in punc:  
            text = text.replace(ele, " ")  
            
    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the the account retweeted from
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub('https', '', text)
    text = re.sub('…', '', text)
    text = re.sub('\u2066', '', text)
    text = re.sub('\u2069 ', '', text)
    text = re.sub(r'\s*<br\s*/?>\s*', u'\n', text)  # newline after a <br>
    text = re.sub(r'</(div)\s*>\s*', u'\n', text)  # newline after </p> and </div> and <h1/>...
    text = re.sub(r'</(p|h\d)\s*>\s*', u'\n\n', text)  # newline after </p> and </div> and <h1/>...
    text = re.sub(r'\s+', u' ', text)  # replace consecutive spaces
    text = re.sub(r'^\s+', u'', text)  # remove spaces at the beginning
    text = re.sub(r'\s+$', u'', text)  # remove spaces at the end
  
    

        
    return text
```

### Import packages and instantiate 
``` python
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
```

#
``` python
df['scores'] = df['full_text'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0.05 else ('neutral' if (c < 0.05 and c > -0.05) else 'neg'))




### Top 5 positive tweets from original content

    He makes me laugh for sure  He s definitely made us a proud people again  I love our country and this is definitely the Trump era realDonaldTrump Trump2020 WalkAway

    JoeBiden  ThePathToSaveAmerica Goes through JoeBiden It is the ONLY WAY to SAVE AMERICA TO SAVE LIVES OF AMERICANS TO SAVE THE CONSTITUTION TO SAVE JOBS

    ewarren is far amp above the best pick She has a grassroots base in place  Her PHD in bankruptcy law makes wall street shudder she is the best most prolific policy wonk  She is a practical yet a kind compassionate soul 

    VP JoeBiden please talk about how you will protect Social Security during DNC2020 This is a top issue for voters 50+ Social Security is a hard earned benefit and a promise that must be kept   ProtectVoters50Plus

    JoeBiden as a White Middleclass woman i beg you to chose a Black woman as your running mate We need it now more than ever There are so many wonderful strong smart Black women who could help save this hurting country

### Top 5 negative tweets from original content 

    realDonaldTrump Maybe the root of evil is drugs  Drugs distort the mind into thinking there is no other way Then sets the place for prostitution robbing stealing murder Every kind of evil gives way to a life of misery and chaos  Which causes destruction and no responsibility 

    How incredibly and unbelievably stupid are some Americans who are lied to every day by realDonaldTrump  who ignore his epic failures who over look his hate and still want to vote for him You people are fools and must be brainless 

    JoeBiden when do we stand up to Trump He and his fascists keep doing all this until WE MAKE THEM STOP  Steal the election if we let them Buddies w 🇷🇺 if we let them What breaks Fascism All the outrages we have seen lived and continue = Fascism  Trump is a Fascist 

    realDonaldTrump if I ever meet u I m Ganna grab your dick so hard and if I get arrested I m getting a fake Russian identity u fuck nachosarah

    Sadly due to CFIX inability to stop violating my rights I will retweet this tweet everyday until Election Day Until the day I’m a second class citizen no more until the day when the confederate stasi no longer exists I will retweet this tweet GoodTrouble Nov3rdIsComing

### Word Cloud

![Alternate image text](/images/twitter/word cloud august.png)