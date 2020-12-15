---
layout: post
title: Stemming, Stop Words, and Text Vectorization. Oh My!
---
![Alternate image text](/images/twitter/books.jpg)

# Part 3: Preparing text data for Topic Modelling

I remember in high school English class looking at individual sentences and breaking it down into multiple ways. With a sentence written on the board, we identified part of speech for each word in order to understand how the words are applied in that sentence. We identified significant words and created meaning with those words, and we identified an emotion or underlining message when present in that sentence. Breaking down a sentence creates meaning. 

Natural Language Processing is part of the machine learning/AI pipeline, where a variety of tasks are applied in in order to process the text data and format it in a way so that the computer can read the data and perform analysis. 

![Alternate image text](/images/twitter/linguistics.png)

I love this figure which is taken from the textbook "Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems" by Vajjala, S. et al. 2020. The image shows the building blocks of a language and in result how NLP is utilized in order to process text data. Within the NLP packages, we can identify meaning through topic modelling and sentiment analysis. We can as well identify syntax through parsing words, morhpemes and lexemes through utilizing natural language tools such as tokenizing, word embeddings, and part of speach tagging, and identify speech and sounds from NLP applications such as speech to text, speaker identification, and text to speech. All of these NLP tools are important to think about when developing machine learning models.

Text creates understanding and explanation about the world around us and we can use text to express how we feel and share our opinion towards a subject. 

*In this blog post I will discuss the different approaches for pre-processing text so that the text is ready to process for machine learning and rule-based algorithms. This blog post is Part 3 in a series of posts in regard to [collecting twitter data on the US Presidential Election](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.*

## Stemming and Lemmetization

``` python
# Import NLTK
import nltk
nltk.download('wordnet')

#import stem package
from nltk import stem
from nltk.corpus import wordnet

# import lemmitzer package
from nltk.stem import WordNetLemmatizer 
```

``` python
# stemmer
porter = stem.porter.PorterStemmer()

# lemmitzation
lem = stem.WordNetLemmatizer()
```

``` python
# Word Lists
word_list1 = ['play', 'playing', 'played']
word_list2 = ['feet', 'foot', 'foots', 'footing']
word_list3 = ['organize', 'organizing', 'organization']
word_list4 = ['benefactor', 'benevolent', 'beneficial']

# Apply Stemmer to word listts
print("['play', 'playing', 'played'] -------------------->", [porter.stem(word) for word in word_list1])
print("['feet', 'foot', 'foots', 'footing'] -------------> ", [porter.stem(word) for word in word_list2])
print("['organize', 'organizing', 'organization'] -------> ", [porter.stem(word) for word in word_list3])
print("['benefactor', 'benevolent', 'beneficial'] -------> ", [porter.stem(word) for word in word_list4])

# Apply Lemmatizer to word lists
print("['play', 'playing', 'played'] -------------------->", [lem.lemmatize(word) for word in word_list1])
print("['feet', 'foot', 'foots', 'footing'] -------------> ", [lem.lemmatize(word) for word in word_list2])
print("['organize', 'organizing', 'organization'] -------> ", [lem.lemmatize(word) for word in word_list3])
print("['benefactor', 'benevolent', 'beneficial'] -------> ", [lem.lemmatize(word) for word in word_list4])
```

**Stemming:** <br>
['play', 'playing', 'played'] --------------------> ['play', 'play', 'play'] <br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['feet', 'foot', 'foot', 'foot'] <br>
['organize', 'organizing', 'organization'] ------->  ['organ', 'organ', 'organ'] <br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevol', 'benefici'] <br>

**Lemmatizer:** <br>
['play', 'playing', 'played'] --------------------> ['play', 'playing', 'played'] <br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['foot', 'foot', 'foot', 'footing'] <br>
['organize', 'organizing', 'organization'] ------->  ['organize', 'organizing', 'organization'] <br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevolent', 'beneficial'] <br>

## Stop Words
1. NLTK
2. SKLEARN
3. Spacy 



References:
[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"} 

Thomas, Rachel. “Topic Modeling with SVD & NMF (NLP Video 2).” YouTube, 8 July 2019, www.youtube.com/watch?v=tG3pUwmGjsc. 

