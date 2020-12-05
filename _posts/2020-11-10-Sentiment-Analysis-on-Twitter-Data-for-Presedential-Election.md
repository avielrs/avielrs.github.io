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

In order to improve the accuracy when processing the tweet data with TextBlob, I first clean the text data by:
 - change uppercase letters to lowercase
 - remove punctuation and emojis
 - remove consecutive spaces
 - remove hyperlinks
 - remove newlines
 - emove retweet account when tweets are retweeted.

#### Use Regex to clean the data:

``` python
# Create a function to clean text
def cleanTxt(text):

    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the the account retweeted from
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub(r'[^\w\s]', '', text) # removes punctuation and emojis
    text = re.sub(r'\s*<br\s*/?>\s*', '\n', text)  # newline after a <br>
    text = re.sub(r'\s+', ' ', text)  # replace consecutive spaces
    text = re.sub(r'^\s+', '', text)  # remove spaces at the beginning
    text = re.sub(r'\s+$', '', text)  # remove spaces at the end
    
    #  make all text lower case
    text = text.lower()
        
    return text
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

After cleaning the text and applying TextBlob Sentiment Analysis the dataframe looks something like this! 
![Alternate image text](/images/twitter/TextBlob_Clean_head5.png)

#### Step 6: Compare sentiment score before cleaning the data with after cleaning the data:

I would like to check to see how much changed the Polarity Scores and Sentiment Analysis between the clean  text and clean text. For me, this is a sanity check to make sure there acutally is a difference after cleaning the text dataset, rather than blindly saying cleaning "Text data will change the outcome of the sentiment score. Awesome!" <br>

To do this, I calculate how many tweets changed its polarity scores after cleaning the text data.
![Alternate image text](/images/twitter/change_unchange_polarity_textblob.png)

**17% percent of the text changed its polarity score after cleaning the text**

It is important to note that even if cleaning text may not seem significant, it  is important to include for setting up the data because text data from tweets inicludes html links, extra space and lines, punctuation, and emojis which all may hinder TextBlob Sentiment Analysis scores.

## VADER SENTIMENT ANLAYSIS
![Alternate image text](/images/twitter/social media sign.jpg)

While TextBlob Sentiment Analysis is a great tool to use, identifying the sentiment in text from social media adds an extra level of complexity compared to identifying sentiment within reviews, online news articles, or books. Social media text is complex because there are emojis to express feelings, acronyms (LOL OMG LMAO ROFL WTF ASAP), intentionally misspelled words like sucks -> sux and fav -> favorite. There are as well slang words that are used on social media that is not identified in the dictionary at least yet such as yolo, muah, haha, woohoo, and using punctuation to make an emotion or face such as (: (; <3 . We as well utilize words in different context. For example, on social media, one might use the word wicked to mean cool/awesome which is takes an originally negative word and utilize it in a positive way. These complexities in social media need to be accounted for when identifying sentiment in a sentence. Luckily for us, computer scientists from Georgeo Tech, C.J. Hutto and Eric Gilbert developed a package called [Vader Sentiment Analysis](https://github.com/cjhutto/vaderSentiment){:target="_blank"} which takes into account many of the edge cases found in social media.

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

#### Step 1: Install Packages and instantiate 
``` python
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
```

#### Step 2: Clean Text
This time when cleaning the text, I do not change all letters to lowercase. This is because vader sentiment analysis takes into account uppercase letters.  Speficially if a word is all capitilize which suggests emphaisis on that word. For example: HURRAY! WIN! FAIL!

This time I am specifically input which punctuation to remove from the text. I decided to exclude exclamation marks ! because Vader Sentiment Anaalysis as well takes into excalamation marks.  

``` python
# Create a function to clean the tweets          
def cleanTxt(text):

    #initializing punctuations string  
    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the the account retweeted from
    # Remove hyperlinks before punctuation is important
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
   
   # define punctuation
    punc = '''()-[]{};:'"\, <>./?@#$%^&*_~'''
  
    # Removing punctuations in string 
    # Using loop + punctuation string 
    for ele in text:  
        if ele in punc:  
            text = text.replace(ele, " ")  
            
    # One last clean with removing spacing
    # After removing punctuation, consecutive spacing may be present
    text = re.sub(r'\s*<br\s*/?>\s*', u'\n', text)  # newline after a <br>
    text = re.sub(r'\s+', u' ', text)  # replace consecutive spaces
    text = re.sub(r'^\s+', u'', text)  # remove spaces at the beginning
    text = re.sub(r'\s+$', u'', text)  # remove spaces at the end
        
    return text
```

#### Step 3: Perform Vader Sentiment Analysis
``` python
df['scores'] = df['full_text'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0.05 else ('neutral' if (c < 0.05 and c > -0.05) else 'neg'))
```

Check out the table after applying Vader Sentiment Analysis!
![Alternate image text](/images/twitter/vsa_Clean_head5.png)

#### Step 4: Sanity Check on cleaning data
Let's do a another sanity check and compare how  many tweets changed its compound score before and after cleaning the text data.
![Alternate image text](/images/twitter/change_unchange_vsa.png)

**12% percent of the text changed its compound score after cleaning the text**

#### Step 5: Compare distirubtion of tweets for negative, neatural, and positive sentiment in August

![Alternate image text](/images/twitter/neg_neut_pos.png)

*42.6% of tweets in August contain positive sentiment (2351 tweets)*
*38.8% of tweets in August contain negative sentiment (2142 tweets)*
*18.6% of tweets in August contain Neutral sentiment (1025 tweets)*

I want to note that this analysis is a sample of tweets taken from August representing a much larger dataset (explained in previous blog post). 


### Top 5 positive tweets from August sample

Win win win win win F🤬k everything else Win win win win WIN

Dear realDonaldTrump I m 68 years old I was a Republican but now I m doing EVERYTHING I can to make sure JoeBiden wins in November YOU WILL NOT STEAL MY SOCIAL SECURITY BENEFITS amp YOU WILL NOT KILL MY POSTAL SERVICE UNDER UNDER ANY CIRCUMSTANCES Sincerely Grandma Grit

Today it was my great honor to proudly accept the endorsement of the NYCPBA! I have deeply and profoundly admired the brave men and women of the NYPD for my entire life New York’s Finest are truly the best of the best — I will NEVER let you down! MAGA

KellyannePolls GOD BLESS YOU🇺🇸 Sweet KellyannePolls U SHINED LIKE A BRIGHT MORNING STAR ⭐️ May The Lord Bless You amp Keep You In His Care🛡 We Will Be Praying For You🙏🏻 You SPOKE The Truth For Millions To Hear And We Know♥️🎚♥️ Thank You Our Love For You Is Great realDonaldTrump🩸 GODWINS

Dear Mr President As you honor and celebrate the life of your best friend and brother Robert please know that millions of Americans are praying that you and your family feel the Presence of God and the Peace that only He can bring realDonaldTrump DonaldJTrumpJr EricTrump

### Top 5 negative tweets from August sample

Pissed off about schools Blame Trump Pissed off about college football Blame Trump Pissed off about 165 000 dead Blame Trump Pissed off about an economy still on its back Blame Trump He didn’t listen didn’t prepare didn’t lead He lied He fucked up Blame him

realDonaldTrump Maybe the root of evil is drugs Drugs distort the mind into thinking there is no other way Then sets the place for prostitution robbing stealing murder Every kind of evil gives way to a life of misery and chaos Which causes destruction and no responsibility

atlpackfan2 onlytruthhere realDonaldTrump Take “Christian” off your profile It’s profane vain for hate intolerance racism pride falsewitness lying resentment impatience cruelty idolatry warmongering arrogance cheating amp antiChrist attitudes acts to rule your heart mind amp life Look to yourself!

Seriously how can anyone with half a functioning brain say JoeBiden is a danger to America when we have the most self serving moronic grifting misogynistic ugly as fuck no heart no brain uncaring constitution killing pussy grabbing punk ass bitch as president now

realDonaldTrump Tucker Carlson is a racist and a brat that they gave a platform to so he can spew his racism OBAMA and the other presidents did the right thing dump trump did the wrong thing as usual Carlson and trump 2 pieces of crap sad sad miserable people you are go get a real life racist

### Word Cloud

![Alternate image text](/images/twitter/word cloud august.png)

### What's next on the agenda?
![Alternate image text](/images/twitter/next.jpg)

Ok, I have now cleaned the the text data, applied Vader Sentiment Analysis to label the sentiment for each tweet. What's next on the agenda? 

Next stop Top Modelling! Well sort of. First I will need to prepare the text data even more before running the topic models which will be discussed in my next blog post 😉. 

With the combination of Topic Modelling and Sentiment Analysis, I will be able to start to form a story of the topics discussed and majority of feeling towards that subject throughout the election. 

