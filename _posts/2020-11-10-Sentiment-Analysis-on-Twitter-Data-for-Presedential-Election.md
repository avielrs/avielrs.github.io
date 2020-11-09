---
layout: post
title: Rule-Based Sentiment Analysis on Twitter Data
---
![Alternate image text](/images/twitter/social_media_sen.jpg)

# Part 2: Sentiment Analysis 

With an abundance of social media, news articles, and opinion pieces available on the web, Data Scientists are able to utilize these resources in order to extract millions of text data. In this project, I am specifically looking at tweets from the 2020 United States Presedential Election. To read how I collected the tweets, check out my previous [post](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}! Tweet data is very advantagous because people use twitter to express opninions and engage with others publicly. From twitter, Data Scientists are able to find key words used amoungst tweets, analyze geographic differences in opinions about topics, detect bots on twitter, analyze user engagment for specific topics, and create a timeline of events from tweet data (Chen et al., 2020). Tweet data can even be utilized to make predictions such as the [2020 US Presedential Elections](https://www.independent.co.uk/news/world/americas/us-election-2020/2020-election-whos-going-to-win-ai-trump-biden-results-outcome-odds-b1374290.html){}:target="_blank"}! Pretty amazing, right? We as well know that social media influences opinions. I have personally experienced this myself by following people who tend to have a certain view point who then shares resources, links, and more education on that view which then makes my opninion even stronger towards that topic. The spread and influence of opinions on social media can be positive and negatively impacted. A rise in hate groups. 

## Purpose
As we can see understanding the intricies of tweet data is greatly beneifical in a variety of ways. For this project, I aam focusing on the context of popular topics from the collected data regarding the 2020 US Presedential election from June 2020 - November 2020, by applying sentiment analysis and topic modelling. **Sentiment Analysis** specifically is a Natural Language Process in order to detect and analyze opinions or attitude within tweets. 

A few example questions to answer with sentiment analysis and topic modelling: 

- How do the majority of tweets collected perceive Trump in total versus for each state? Does the perception change from June 2020 to October 2020?
- How do the majority of tweets collected perceive Biden in total versus for each state? Does the perception change from June 2020 to October 2020?
- How do the majority of tweets collected perceive Kamala Harris in total versus for each state?
- How do the majority of tweets collected perceive COVID19 in total versus for each state?
- How do the majority of tweets collected perceive voter fraud in total versus for each state? 
- How do tweets with a large following perceive certain topics?

### Methods

#### Rule Based
Describe what it is
- Text Blob
- Vader Sentiment Analysis

#### Machine Learaning
 Describe what it is
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

From TextBlob, the sentiment returns a polarity and subjectivity score. The polarity score output is a float between the range [-1.0, 1.0], where -1.0 is 100% negativea and 1.0 is 100% positive. The subjectivity is a float within the range [0.0, 1.0] where 1.0 is very subjective (influence by a personal feeling or opinion) and 0.0 is very objective (not influenced by personal feelings or opinion).

#### Step 1: Import libraries
``` python
from textblob import TextBlob
```

#### Step 2: Clean text data 

Let’s take a look at a tweet example from the dataset: 
	  
![Alternate image text](/images/twitter/original_tweet.png)

The first thing to note from this tweet is that the subjectivity is positive and will probably be ranked near 1. The other thing to note is that this is a retweet (RT) from tweet user Kayleigh McEnany. 

Things to note about the text:
1.	There are breaks (newlines) in this tweet
2.	Punctuation: ‘@’,’!’, ‘…’, ‘:’, ‘/’, ‘.’
3.	Html link is present
4.	Arrow Emoji 

In order to improve the accuracy when processing the tweet data with TextBlob, I first clean the text data by changing uppercase letters to lowercase, removing RT and the @username associated with the RT (retweet), remove hyperlinks, remove punctuation and emojis, remove consecutive spaces, remove breaks, remove extra spaces at the beginning and end of the tweet. 

#### Use Regex to clean the data:

``` python

def cleanTxt(text):
        
    text = re.sub('RT[\s]@[A-Za-z0–9]+', '', text) # Removing RT and the the account retweeted from
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

After cleaning the text, the tweet now looks like this:
![Alternate image text](/images/twitter/clean_texts.png)

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

#### Step 5: Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
``` python

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
```

We can also take a quick look
![Alternate image text](/images/twitter/text_blob_august_sentiment.png)

#### Step 6: View Analysis and compare with non-cleaned text data

Without Punctation
Subjectivity: 0.45
Polarity: 0.16
Analysis: Positive

With Punctuation
Subjectivity =  0.43
Polarity = 0.47
Analysis: Positive
Both are subjective, however the Polarity with the punctation is less positive then the polarity without the punctuation. One thing to note, is that while the TextBlob analysis rated the tweet as subjective. It is rated as 0.45 between (0-1). Because of the explaination marks and 

## Vader Analysis
![Alternate image text](/images/twitter/social media sign.jpg)

[Vader Sentiment Analysis](https://github.com/cjhutto/vaderSentiment){:target="_blank"} is a lexicon rule-based sentiment analysis tool that was specifically developed for social media text. Vader Sentiment accounts for speed and performance which is important for large datasets such as thousands or millions of tweets.

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

The sentiment score of a text can be obtained by summing up the intensity of each word in the text.
    positive sentiment: compound score >= 0.05
    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    negative sentiment: compound score <= -0.05

``` python
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
```

``` python
df['scores'] = df['full_text'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0.05 else ('neutral' if (c < 0.05 and c > -0.05) else 'neg'))
```



### Word Cloud Fun

``` python
    #remove stop words
    import spacy

    spacy.prefer_gpu()
    spacy_nlp = spacy.load("en_core_web_sm")

    # Stop words from spacy
    all_stopwords = nlp.Defaults.stop_words


    comment_words = '' 
    
    # iterate through the csv file 
    for val in Text: 
        
        # typecaste each val to string 
        val = str(val) 
    
        # split the value 
        tokens = val.split() 
        
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 1000, height = 600, 
                    background_color = 'white',
                # colormap = 'Spectral',
                    stopwords = all_stopwords, 
                    min_font_size = 10).generate(comment_words) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (10, 6), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show()

```
![Alternate image text](/images/twitter/word_cloud_august_1.png)


#### References

Chen E, Lerman K, Ferrara E
Tracking Social Media Discourse About the COVID-19 Pandemic: Development of a Public Coronavirus Twitter Data Set
JMIR Public Health Surveill 2020;6(2):e19273
URL: https://publichealth.jmir.org/2020/2/e19273
DOI: 10.2196/19273
PMID: 32427106
PMCID: 7265654

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.