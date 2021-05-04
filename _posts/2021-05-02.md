---
layout: post
title: Part 1 Stemming vs Lemmatization
---
![Alternate image text](/images/twitter/books.jpg)
 
# Part 3A: Text Preprocessing for Topic Modelling
 
I love the word Lemmatization because the syllables, consonants, and vowels that make up the word lemmatization form a sound that makes me smile. I like Natural Language Processing (NLP) because I like to think about languages as a structure, pattern, and rhythm. Through this process, I can take a sentence and visualize it. Like a fun puzzle, languages are messy at first, but a comprehensive picture forms when the right pieces fit together.
 
I remember sitting in my high school English literature class with multiple sentences written on the board. We broke down each sentence and identified part of speech to understand how a word’s structure influences a sentence’s meaning. The act of breaking down a sentence to understand the structure falls under the discipline of linguistics.  While as individuals, we can break down sentences and identify meaning, it becomes difficult and almost impossible to do so with large text documents such as 100,000,000 tweets or 100,000 news articles. <br> <br>
 
![Alternate image text](/images/twitter/linguistics.png)
 
I love this figure taken from the textbook *"Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems" by Vajjala, S. et al. 2020.* The image shows that context, syntax, morphemes and lexemes, and phonemes are the language blocks. A variety of packages from NLTK, Spacy, TextBlob, and more, are Natural Language Processing applications that identify each language block. For example, to determine the context (meaning) with NLP, we can apply Topic Modelling and Sentiment Analysis.
 
#### Definition of Natural Language Processing
Natural Language Processing is part of the machine learning/AI pipeline. A variety of tasks are applied to process text data and format it so that the computer can read the data and perform analysis.
 
*In this blog post I will discuss stemming and lemmatization, a pre-processing method for text data so that the text is ready to process for machine learning and rule-based algorithms. Specifically, in regards to topic modelling with the use of Twitter text data. This blog post is Part 3 in a series of posts in regard to [collecting twitter data on the US Presidential Election](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.*
 
### Why use stemming and lemmatization?
Each document contains a vector of words (terms). Sentence tokenization separates each word into a matrix where each term is a feature. For example, if a sentence (Document 1) contains the term **sit**, and Document 2 has the term **sitting**. The two terms will end up as two separate column features even though the meaning is the same. However, if we reduce the word sitting to its root word sit, then the document matrix is reduced.
 
![Alternate image text](/images/twitter/diagram_lem_stem_token.png)
 
Stemming and Lemmatization are two different approaches for stripping a term within a document so that a document matrix reduces and the complexity of data decreases. Reducing the size and complexity of a model helps achieve model accuracy and reduce computation memory and time.
 
#  Stemming
![Alternate image text](/images/twitter/stemming.jpg)
 
**Stemming** is a text processing method in which a term reduces to its "stem" or simplest form by removing suffixes from the term such as (-ED, -ING, -ION, -IONS, -S). Suffixes are removed specifically for IR performance, not for linguistic meaning (Porter, 1980).
 
**Types of Stemming:**
1. Porter Stemmer
2. Lovinus Stemmer
3. Paice Stemmer
 
Stemming removes case letters at the end of a word. For example, when applying stemming to the word *catastrophe* the 'e' at the end of the word is removed.
 
catastroph**e** --> catostroph
 
**Significance of stemming:** <br>
Stemming can be beneficial when developing search engines for query matching and feature space reduction when training a model (B., Sowmya V et al., 2020).  
 
Example rules of Porter Stemmer:
 
SSES ---> SS <br>
care**sses** ---> caress<br><br>
IES ---> I <br>
pon**ies** ---> poni<br><br>
SS ---> SS  <br>
care**ss** ---> caress<br><br>
S --->      
cat**s** ---> cat <br><br>
 
Import and instantiate Porter Stemmer:
 
``` python
# Import NLTK
import nltk
nltk.download('wordnet')
 
#import stem package
from nltk import stem
from nltk.corpus import wordnet
 
# stemmer
porter = stem.porter.PorterStemmer()
```
 
Let's see some examples of how Porter Stemmer is applied:
``` python
# Create a word list
word_list1 = ['play', 'playing', 'played']
word_list2 = ['feet', 'foot', 'foots', 'footing']
word_list3 = ['organize', 'organizing', 'organization']
word_list4 = ['benefactor', 'benevolent', 'beneficial']
word_list5 = ['universe', 'university']
 
print("['play', 'playing', 'played'] -------------------->", [porter.stem(word) for word in word_list1])
print("['feet', 'foot', 'foots', 'footing'] -------------> ", [porter.stem(word) for word in word_list2])
print("['organize', 'organizing', 'organization'] -------> ", [porter.stem(word) for word in word_list3])
print("['benefactor', 'benevolent', 'beneficial'] -------> ", [porter.stem(word) for word in word_list4])
print("['universe', 'university'] -------> ", [porter.stem(word) for word in word_list5])
```
**Output:**<br>
['play', 'playing', 'played'] --------------------> ['play', 'play', 'play'] <br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['feet', 'foot', 'foot', 'foot'] <br>
['organize', 'organizing', 'organization'] ------->  ['organ', 'organ', 'organ']<br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevol', 'benefici'] <br>
['universe', 'university'] ------->  ['univers', 'univers'] <br><br>
 
### Pros for stemming
1. Remove sufixes
2. Reduce the size and complexity of data
3. Reduce variance in models which can cause overfitting.
 
### Cons for stemming
Stemming is too good at extracting the root word. One might say that stemming "butchers" the terms in a document. Stemming will take a word like **organize** and shorten it to **organ** which has an entirely different meaning. Same with the word **University**, which will stem to **univers**, again shortening the word to a word that is not the same meaning. When deciding to use Porter Stemmer, determine how important are the words meaning versus reducing the data’s complexity to your model and analysis.
 

 
#### References

[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"}
 
Thomas, Rachel. “Topic Modeling with SVD & NMF (NLP Video 2).” YouTube, 8 July 2019, [www.youtube.com/watch?v=tG3pUwmGjsc](www.youtube.com/watch?v=tG3pUwmGjsc){:target="_blank"}.
 
[https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"}
 
Porter, Martin F. 1980. An algorithm for suffix stripping. Program 14 (3): 130-137.
 
[http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer){:target="_blank"}
 
[http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize){:target="_blank"}
 
 

