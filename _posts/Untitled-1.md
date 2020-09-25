---
layout: post
title: TF-IDF and One Hot Encoding, Oh My!
---

### Part 4: Use NLP to transform text data into numeric values
In order to perform predictive models on the dataset, the text data must first be transformed into numeric values. I used pd.get_dummies which is a one-hot-encoding process to transform Country, Province, and Variety into a 1 or 0. I then used TF-IDF from sklearn to transform the wine descriptions into weighted tokens.

#### Step 1: Use One Hot Encoding to transform Province, Country, and Variety into 1 or 0

Use [get dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) for the columns: Province, Country and Variety. This will transform all text in these columns into 1 or 0

EXAMPLE: 

Country = ['Italy', 'USA', 'Australia', 'Italy']

    df = pd.get_dummies(Country)

|      |    Italy   |     USA   |     Australia   |
|------|:----------:|:---------:|:---------------:|
|  0   |   1   |  0  |    0      |
|  1   |   0   |  1  |    0      |
|  2   |   0   |  0  |    1      |
|  3   |   1   |  0  |    0      |


#### Step 2: TF-IDF Vectorizer

Import [TFIDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) package from Sklearn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```



```python

import string

import nltk
stemmer = nltk.stem.PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords 
ENGLISH_STOP_WORDS = stopwords.words('english')

def my_tokenizer(sentence):
    
    for punctuation_mark in string.punctuation:
        # Remove punctuation and set to lower case
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    
        
    # Remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words 
    
```



```python
# View tokens associated to their weight for the train dataset
word_weights = np.array(np.sum(review_train, axis=0)).reshape((-1,))

words = np.array(tfidf.get_feature_names())
words_df = pd.DataFrame({"word": words, 
                         "weight": word_weights}) 
```
