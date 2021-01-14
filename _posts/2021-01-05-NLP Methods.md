---
layout: post
title: Stemming vs Lemmatization
---
![Alternate image text](/images/twitter/books.jpg)

# Part 3A: Text Preprocessing for Topic Modelling

I love the word 'Lemmatization'. Something about the cumulation of sounds that are formed from the syllables, vowels, and consonants that make up the word 'Lemmatization' makes me smile. The first time I heard the word 'Lemmatization', I knew I needed to learn more about it! I guess that is why I like studying Natural Language Processing (NLP). I am drawn towards the subject because I like to think about languages as a structure, patten, and rhythm. Through this process, I am able to take a sentence and visualize it. Just like a fun puzzle, languages are messy at first but when the right pieces fit together, a clear picture is formed.

I remember sitting in my high school English literature class with multiple sentences written on the board. For the class assingment, we broke down each sentence and identified the part of speech for each word. By doing this, we gained understanding and meaning of how each word was applied in that sentence. The act of breaking down a sentence in order to understand the structure of a sentence falls under the discpline of linguistics.  While as individuals, we can break down sentences and idenify meaning, it becomes difficult and almost impossible to do so with large text documents such as 100,000,000 tweets or 100,000 news articles. This is where natural language processing comes into effect. The variety of packages provided for NLP allows us to process big text data efficently. 

![Alternate image text](/images/twitter/linguistics.png)

I love this figure which is taken from the textbook *"Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems" by Vajjala, S. et al. 2020.* The image shows the building blocks of a language and in result how NLP is utilized in order for computers to process text data. 

#### Definition of Natural Language Processing
Natural Language Processing is part of the machine learning/AI pipeline, where a variety of tasks are applied in order to process text data and format it in a way so that the computer can read the data and perform analysis. 

*In this blog post I will discuss stemming and lematization, a pre-processing method for text data so that the text is ready to process for machine learning and rule-based algorithms. Specifically, in regards to topic modelling with the use of Twitter text data. This blog post is Part 3 in a series of posts in regard to [collecting twitter data on the US Presidential Election](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.*

# Stemming and Lemmatization
Each document contains a vector of words (terms), in this case, the document is the tweet. Sentence tokenization separates each word into a matrix where each term is a feature. For example, if a sentance (or document) contains the term **sit**, and another document contains the term **sitting**. The terms will end up in separate columns even though the meaning is the same. 

![Alternate image text](/images/twitter/diagram_lem_stem_token.png)

This is where stemming and lemmatization may be beneficial. Stemming and Lemmatization are two separate approaches for stripping a term within a document so that a document matrix is reduced and thus the complexity of data decreases. Reducing size and complexity of a model is beneficial for achieving model accuracy and for reducing computationally memory and time.

## Stemming
![Alternate image text](/images/twitter/stemming.jpg)

**Stemming** is a text processing method in which a term is reduced to its "stem" or simplest form through the removal of suffixes from the term such as (-ED, -ING, -ION, -IONS, -S). Suffixes are removed specifically for IR performance, not for linguistic meaning (Porter, 1980).

**Types of Stemming:**
1. Porter Stemmer 
2. Lovinus Stemmer
3. Paice Stemmer

Stemming removes case letters at the end of a word. For example, when applying stemming to the word *catastrophe* the 'e' at the end of the word is removed. 

catastroph**e** --> catostroph 

**Significance of stemming:** <br>
Stemming can be beneficial when developing search engines where a query can be matched and in text classification where feature space can be reduced when training a model (B., Sowmya V et al., 2020).   

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
**output**<br>
['play', 'playing', 'played'] --------------------> ['play', 'play', 'play'] <br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['feet', 'foot', 'foot', 'foot'] <br>
['organize', 'organizing', 'organization'] ------->  ['organ', 'organ', 'organ']<br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevol', 'benefici'] <br>
['universe', 'university'] ------->  ['univers', 'univers'] <br>

**Pros for stemming:**
1. Remove sufixes
2. Reduce size and complexity of data
3. Reduce variance in models which can cause overfitting.

**Cons for stemming:** Stemming does "too" good of a job of extracting the root word or one could say stemming "butchers" the word. For example, 

Stemming will take a word like **organize** and shorten it to **organ** which has an entirely different meaning. Same with the word **University**, which will stem to **univers**, again shortening the word to a word that is not the same meaning. This is something to consider when using Porter Stemmer. How important is the meaning of the word versus reducing the complexity of the data to your model and analysis?

## Lemmatization
![Alternate image text](/images/twitter/dictionary.jpg)

**Lemmatization** is another approach that handles term. Lemmatization labels the term from its base word (lemma). This method is a more methodical approach for ensuring the words are reduced without losing its meaning.

To get a better understanding of how lemma is used within linguestics, let's take a trip down memory lane and recall verb conjugation. 

**Conjucating of 'To Be'**

*Present Tense:*<br>
        I am <br>
        You are <br>
        He/She/It is <br>
        We are <br>
        You are <br>
        They are <br>

*Past Tense:* <br>
        I was <br>
        You were <br>
        He/She/It was <br>
        We were <br>
        You were <br>
        They were <br>

The lemma or lemmatization of AM, IS, ARE, WAS, and were is 'BE'

Code test: 
```python
print('I am ---> To', lemmatizer.lemmatize("am", pos="v")) #v is for verb”
print('You are --> To', lemmatizer.lemmatize("are", pos="v")) #v is for verb”
print('He is --> To', lemmatizer.lemmatize("is", pos="v")) #v is for verb”
print('They were --> To', lemmatizer.lemmatize("were", pos="v")) #v is for verb” 
```
I am ---> To be <br>
You are --> To be <br>
He is --> To be <br>
They were --> To be <br>

**Note:** In lemmatization, the part of speech (pos) needs to be defined. In the example above, I define the pos as "v" for verb. If the pos parameter is not defined, then the default is set to NOUN.

Examples of how Lemmatization is applied: 
```python
print("['play', 'playing', 'played'] -------------------->", [lem.lemmatize(word) for word in word_list1])
print("['feet', 'foot', 'foots', 'footing'] -------------> ", [lem.lemmatize(word) for word in word_list2])
print("['organize', 'organizing', 'organization'] -------> ", [lem.lemmatize(word) for word in word_list3])
print("['benefactor', 'benevolent', 'beneficial'] -------> ", [lem.lemmatize(word) for word in word_list4])
print("['universe', 'university'] -------> ", [lem.lemmatize(word) for word in word_list5])
```
**Output** <br>
['play', 'playing', 'played'] --------------------> ['play', 'playing', 'played']<br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['foot', 'foot', 'foot', 'footing']<br>
['organize', 'organizing', 'organization'] ------->  ['organize', 'organizing', 'organization']<br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevolent', 'beneficial']<br>
['universe', 'university'] ------->  ['universe', 'university'] <br>

Even though part of speech is not identified in the above example, we can see here that Lemmatization is more conservative about trimming a word then in stemming. University does not change to universe and organize/organization does not change to organ.

Lemmatization is great at even identifying the base word for complex verbs. See example below:
```python
print('beheld', lem.lemmatize('beheld', pos = 'v'))
print('witheld', lem.lemmatize('withheld', pos = 'v'))
print('flung', lem.lemmatize('flung', pos = 'v'))
```
beheld behold <br>
witheld withhold <br>
flung fling <br>

Handling plural in lemmatization:
```python
print(lemmatizer.lemmatize("ponies"))
print(lemmatizer.lemmatize("caresses"))
print(lemmatizer.lemmatize("cats"))
```
pony<br>
caress<br>
cat<br>

Identifying part of speech for nouns and adjectives is more meaningful within languages other than English. Basically, languages that utilize grammatical gender. 

#### Example of how defining pos for adjectives is important in Lemmatization
In Hebrew, the word for 'big' is גָּדוֹל (gadol):

![Alternate image text](/images/twitter/hebrew_gadol.png)

In hebrew, the ending of the adjective changes according to if the word is used as singlular masculine, singlular feminine, plural masculine, or plural feminine. The root (lemma) of gadol is  ג - ד - ל (g-d-l).Thus when lemmatization is applied to the hebrew word גָּדוֹל (Gadol), the word will be reduced to its root word גדל (gdl).

**Pros for lemmetization**
1. Using the base word ensures that the meaning behind the word is not being lost

**Cons for lemmetization**
1. Need to identify part of speech
2. Need to understand fundamentals of linguistics thus more complex

#### Spacy Lemmatization 
![Alternate image text](/images/twitter/lem_without_pos.png)

```python

import spacy
sp = spacy.load('en_core_web_sm')
```
![Alternate image text](/images/twitter/spacy_lem.png)

#### References
[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"} 

Thomas, Rachel. “Topic Modeling with SVD & NMF (NLP Video 2).” YouTube, 8 July 2019, [www.youtube.com/watch?v=tG3pUwmGjsc](www.youtube.com/watch?v=tG3pUwmGjsc){:target="_blank"}. 

[https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"}

Porter, Martin F. 1980. An algorithm for suffix stripping. Program 14 (3): 130-137.

[http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer){:target="_blank"}

[http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize){:target="_blank"}
