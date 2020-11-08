---
layout: post
title: Sentiment Analysis towards Twitter Data from the 2020 Presedential Election
---
![Alternate image text](/images/twitter/emojis.jpg)

# Part 2: Sentiment Analysis 

Text is a form of creating understanding and explanation about the world around us. With an abundance of social media, news articles, and opinion pieces available on the web, Data Scientists are able to utilize these resources in order to develop recommender systems, predict text, identify sentiment and major topics within a document, and even predict outcomes such as the [2020 United States Election](https://www.independent.co.uk/news/world/americas/us-election-2020/2020-election-whos-going-to-win-ai-trump-biden-results-outcome-odds-b1374290.html){:target="_blank"}. In order for millions of text data to be processed efficiently, we use a process called Natural Language Processing.

Natural Language Processing is part of the machine learning/AI pipeline, where a variety of tasks are applied in in order to process the text data and format it in a way so that the computer can read the data and perform analysis.

![Alternate image text](/images/twitter/linguistics.png)

I love this figure which is taken from the textbook "Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems" by Vajjala, S. et al. 2020. The image shows the building blocks of a language and in result how NLP is utilized in order to process text data. Within the NLP packages, we can identify meaning through topic modelling and sentiment analysis. We can identify syntax through parsing words. We can identfy morhpemes and lexemes through utilizing natural language tools called tokenizing, word embeddings, and part of speach tagging. We can as well identify speech and sounds through applying speech to text, speaker identification, and text to speech. 

### Purpose

For this project, I aam focusing on context from tweet data through sentiment analysis and topic modelling. **Sentiment Analysis** specifically is a Natural Language Process in order to detect and analyze opinions or attitude within a text or document. To read how I collected the tweets, you can read my previous [post](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.  

A few example questions to answer with sentiment analysis and topic modelling: 

- How do the majority of tweets collected perceive Trump in total versus for each state? Does the perception change from June 2020 to October 2020?
- How do the majority of tweets collected perceive Biden in total versus for each state? Does the perception change from June 2020 to October 2020?
- How do the majority of tweets collected perceive Kamala Harris in total versus for each state?
- How do the majority of tweets collected perceive COVID19 in total versus for each state?
- How do the majority of tweets collected perceive voter fraud in total versus for each state? 
- How do tweets with a large following perceive certain topics?

### Methods

#### 1) Rule Based
- Text Blob
- Vader Sentiment Analysis

#### 2) Machine Learning
- SVM
- Naïve Bayes

### Step 1: Text Blob

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


I will use TextBlob specifically for Sentiment Analysis. The sentiment returns a polarity score as a float between the range [-1.0, 1.0]. Where 1.0 is very subjective (influence by a personal feeling or opinion), 0.0 is very objective (not influenced by personal feelings or opinion).

**Import libraries**
```
from textblob import TextBlob
```

** Create a function to get the subjectivity and polarity**
```
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return  TextBlob(text).sentiment.polarity


# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Full_Text'].apply(getSubjectivity)
df['Polarity'] = df['Full_Text'].apply(getPolarity)
```

```
# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
```

Let’s take a look at a tweet and see how the TextBlob identiefies the tweet: 

	Print(print(df['full_text'][0])

	  
![Alternate image text](/images/twitter/tweet_example_1.png)

The first think to note from this tweet is that the subjectivity is positive and will probably be ranked near 1.0. The other thing to note is that this is retweet (RT) from tweet user Kayleigh McEnany. Next let’s look the setup of the tweet. Things to note about  this  tweet:
1.	There are breaks (newlines) in this tweet
2.	Punctuation: ‘@’,’!’, ‘…’, ‘:’, ‘/’, ‘.’
3.	Html link is present
4.	Arrow Emoji 

In order to improve the accuracy when processing the tweet data with TextBlob, I first clean the text data by changing uppercase letters to lowercase, removing RT and the @username associated with the RT (retweet), remove hyperlinks, remove punctuation and emojis, remove consecutive spaces, remove breaks, remove extra spaces at the beginning and end of the tweet. 


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

	# Clean the tweets
Text = df['full_text'].apply(cleanTxt)

# Add lower cases
for i in np.arange(0, len(Text), 1):
    Text[i] = Text[i].lower()

After cleaning the text data, the tweet looks likes this: 

overflow crowd for president realdonaldtrump in new hampshire we are going to win via abatemedia

Without Punctation
Subjectivity: 0.45
Polarity: 0.16
Analysis: Positive

With Punctuation
Subjectivity =  0.43
Polarity = 0.47
Analysis: Positive
Both are subjective, however the Polarity with the punctation is less positive then the polarity without the punctuation. One thing to note, is that while the TextBlob analysis rated the tweet as subjective. It is rated as 0.45 between (0-1). Because of the explaination marks and 
![Alternate image text](/images/twitter/text_blob_august_sentiment.png)


Word Cloud

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

![Alternate image text](/images/twitter/word_cloud_august_1.png)


#### References

[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"} 

