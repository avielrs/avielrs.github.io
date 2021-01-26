---
layout: post
title: Stemming vs Lemmatization
---
![Alternate image text](/images/twitter/books.jpg)

# Part 3A: Text Preprocessing for Topic Modelling

I love the word 'Lemmatization'. Something about the cumulation of sounds that are formed from the syllables, vowels, and consonants that makes up the word 'Lemmatization' makes me smile. The first time I heard the word 'Lemmatization', I knew I needed to learn more about it! I guess that is why I like studying Natural Language Processing (NLP). I am drawn to the subject because I like to think about languages as a structure, patten, and rhythm. Through this process, I am able to take a sentence and visualize it. Just like a fun puzzle, languages are messy at first but when the right pieces fit together, a clear picture is formed.

I remember sitting in my high school English literature class with multiple sentences written on the board. For the class assignment, we broke down each sentence and identified the part of speech for each word. By doing this, we gained understanding and meaning of how each word was applied in that sentence. The act of breaking down a sentence in order to understand the structure falls under the discipline of linguistics.  While as individuals, we can break down sentences and identify meaning, it becomes difficult and almost impossible to do so with large text documents such as 100,000,000 tweets or 100,000 news articles. This is where natural language processing can help, and the variety of packages provided for NLP allows us to process big text data efficiently. <br> <br>

![Alternate image text](/images/twitter/linguistics.png)

I love this figure which is taken from the textbook *"Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems" by Vajjala, S. et al. 2020.* The image shows that a language is made up of fundmental blocks that can be broken down in order to gain a greater understanding of a sentence. Each block can be processed with an NLP application. For example, in order to identify context (meaning) with NLP, we can apply Topic Modelling and Sentiment Analysis. 

#### Definition of Natural Language Processing
Natural Language Processing is part of the machine learning/AI pipeline, where a variety of tasks are applied in order to process text data and format it in a way so that the computer can read the data and perform analysis. 

*In this blog post I will discuss stemming and lematization, a pre-processing method for text data so that the text is ready to process for machine learning and rule-based algorithms. Specifically, in regards to topic modelling with the use of Twitter text data. This blog post is Part 3 in a series of posts in regard to [collecting twitter data on the US Presidential Election](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.*

### Why stemming and lemmatization is used? 
Each document contains a vector of words (terms). Sentence tokenization separates each word into a matrix where each term is a feature. For example, if a sentence (or document) contains the term **sit**, and another document contains the term **sitting**. The two terms will end up as two separate column features even though the meaning is the same. 

![Alternate image text](/images/twitter/diagram_lem_stem_token.png)

This is where stemming and lemmatization may be beneficial. Stemming and Lemmatization are two separate approaches for stripping a term within a document so that a document matrix is reduced and thus the complexity of data decreases. Reducing size and complexity of a model is beneficial for achieving model accuracy and for reducing computation memory and time.

# I. Stemming
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
**Output:**<br>
['play', 'playing', 'played'] --------------------> ['play', 'play', 'play'] <br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['feet', 'foot', 'foot', 'foot'] <br>
['organize', 'organizing', 'organization'] ------->  ['organ', 'organ', 'organ']<br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevol', 'benefici'] <br>
['universe', 'university'] ------->  ['univers', 'univers'] <br><br>

### Pros for stemming
1. Remove sufixes
2. Reduce size and complexity of data
3. Reduce variance in models which can cause overfitting.

### Cons for stemming
Stemming does "too" good of a job of extracting the root word or one could say stemming "butchers" the word. Stemming will take a word like **organize** and shorten it to **organ** which has an entirely different meaning. Same with the word **University**, which will stem to **univers**, again shortening the word to a word that is not the same meaning. This is something to consider when using Porter Stemmer. How important is the meaning of the word versus reducing the complexity of the data to your model and analysis?

# II. Lemmatization
![Alternate image text](/images/twitter/dictionary.jpg)

**Lemmatization** labels the term from its base word (lemma). This method is a more methodical approach for ensuring the words are reduced without losing its meaning.

## A. NLTK Lemmatization
```python
# import lemmatizer package
from nltk.stem import WordNetLemmatizer 
```
Examples of how Lemmatization is applied: 
```python
print("['play', 'playing', 'played'] -------------------->", [lem.lemmatize(word) for word in word_list1])
print("['feet', 'foot', 'foots', 'footing'] -------------> ", [lem.lemmatize(word) for word in word_list2])
print("['organize', 'organizing', 'organization'] -------> ", [lem.lemmatize(word) for word in word_list3])
print("['benefactor', 'benevolent', 'beneficial'] -------> ", [lem.lemmatize(word) for word in word_list4])
print("['universe', 'university'] -------> ", [lem.lemmatize(word) for word in word_list5])
```
**Output:** <br>
['play', 'playing', 'played'] --------------------> ['play', 'playing', 'played']<br>
['feet', 'foot', 'foots', 'footing'] ------------->  ['foot', 'foot', 'foot', 'footing']<br>
['organize', 'organizing', 'organization'] ------->  ['organize', 'organizing', 'organization']<br>
['benefactor', 'benevolent', 'beneficial'] ------->  ['benefactor', 'benevolent', 'beneficial']<br>
['universe', 'university'] ------->  ['universe', 'university'] <br>

We can see here that Lemmatization is more conservative about trimming a word then in stemming. University does not change to universe and organize/organization does not change to organ.<br><br>

### Handling plural in lemmatization
```python
lemmatizer.lemmatize("ponies")
lemmatizer.lemmatize("caresses")
lemmatizer.lemmatize("cats")
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ponies ---> pony<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; caresses ---> caress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats ---> cat<br><br>

### Understanding Lemma
To gain a better understanding of how lemma is used within linguistics, let's take a trip down memory lane and recall verb conjugation. 

**Conjucate 'To Be'**

&nbsp;&nbsp; Present Tense:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  I am <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  You are <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  He/She/It is <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We are <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  You are <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  They are <br>

&nbsp;&nbsp; Past Tense: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  I was <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  You were <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  He/She/It was <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We were <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  You were <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  They were <br>

Thus, the lemma or lemmatization of AM, IS, ARE, WAS, and were is 'BE'

Code test: 
```python
print('I am ---> To', lemmatizer.lemmatize("am", pos="v")) #v is for verb”
print('You are --> To', lemmatizer.lemmatize("are", pos="v")) #v is for verb”
print('He is --> To', lemmatizer.lemmatize("is", pos="v")) #v is for verb”
print('They were --> To', lemmatizer.lemmatize("were", pos="v")) #v is for verb” 
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; I am ---> To be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You are --> To be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; He is --> To be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; They were --> To be <br>

**Note:** In lemmatization, the part of speech (pos) needs to be defined. In the example above, I define the pos as "v" for verb. If the pos parameter is not defined, then the default is set to NOUN.<br>

### Complex verbs in NLTK Lemmatizer
Has the capability to identify base words from complex verbs. See example below:
```python
lem.lemmatize('beheld', pos = 'v')
lem.lemmatize('withheld', pos = 'v')
lem.lemmatize('flung', pos = 'v')
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; beheld ---> behold <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; witheld ---> withhold <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; flung ---> fling <br> <br>

### Adverbs in NLTK Lemmatizer

```python
# adverb
print('farther', lem.lemmatize('farther', pos = 'r'))
# superlative adverb
print('farthest', lem.lemmatize('farthest', pos = 'r'))

# adverb
print('loudly', lem.lemmatize('loudly', pos = 'r'))
# superlative adverb
print('loudest', lem.lemmatize('loudest', pos = 'r'))
```
**Output** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farther ---> far <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farthest ---> farthest <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; loudly ---> loudly <br>

NLTK Lemmatizer can handle basic adverbs such as the words farther and loudly. However, adverbs that end in 'est', also known as superlative adverbs, is not supported by NLTK Lemmatizer. Therefore, loudest does not change to loud and farthest does not change to far.

### Adjectives with different endings

```python
# comparative adjective
print('closer', lem.lemmatize('closer', pos = 'a'))

#superlative adjective
print('closest', lem.lemmatize('closest', pos = 'a'))

# comparative adjective:
print('smaller', lem.lemmatize('smaller', pos = 'a'))

# superlative adjective
print('smallest', lem.lemmatize('smallest', pos = 'a'))

# Dry
print('drier', lem.lemmatize('drier', pos = 'a'))
print('driest', lem.lemmatize('driest', pos = 'a'))
```

**Output**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; closer close <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; closest close <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; smaller small <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; smallest small <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; drier dry <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; driest dry <br>

NLTK Lemmatizer handles adjectives with different endings very well.

### Part of Speech (POS)
In English, I have demonstrated from above that there are different endings for adjectives such as smaller and smallest, and different endings for adverbs such as farther and loudly. I as well showed the significance of the lemmatizer for verb conjugation. The English language while does have some different ending types to words, the ending of a word within nouns and adjectives are more meaningful within languages other than English. Basically, languages that utilize grammatical gender. 

For example, in Hebrew, the word for 'big' is גָּדוֹל (gadol). <br>

![Alternate image text](/images/twitter/hebrew_gadol.png)

In Hebrew, the ending of the adjective changes according to if the word is used as singular masculine, singular feminine, plural masculine, or plural feminine. The root (lemma) of gadol is  ג - ד - ל (g-d-l).Thus when lemmatization is applied to the Hebrew word גָּדוֹל (Gadol), the word is reduced to its root word גדל (gdl). <br><br>

### Pros for lemmatization
1. Using the base word ensures that the meaning behind the word is not lost

### Cons for lemmatization
1. Need to identify part of speech
2. Need to understand fundamentals of linguistics thus more complex

## B. Spacy Lemmatization 
Spacy provides its own lemmatization package. Let's see how this package compares with the lemmatizer in NLTK. 

### Spacy Lemmatization
In spacy lemmatization, part of speech for each word is not a parameter.

```python
# import spacy package
import spacy

# load English spacy
sp = spacy.load('en_core_web_sm')

# create word list
words = ['better','ran', 'are', 'running', 'were', 'shared', 'organize', 'university', 
         'awoken', 'arose', 'beheld', 'sped', 'withhold', 'flung', 'cats', 'timely', 'actively', 'tighter', 'smaller', 'farther', 'driest', 'farthest', 'loudly']

for i in words:
    token = sp(i)
    for word in token:
        print(word.text,'-->',  word.lemma_)
```
**Output:** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; better --> well <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ran --> run <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; are --> be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; running --> run <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; were --> be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shared --> share <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; organize --> organize <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; university --> university <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; awoken --> awake <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; arose --> arise <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; beheld --> beheld <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; sped --> speed <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; withhold --> withhold <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; flung --> flung <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats --> cat <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; timely --> timely <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; actively --> actively <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tighter --> tight <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; smaller --> small <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farther --> far <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; driest --> dry <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farthest --> farthest <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loudly --> loudly <br>

### NLTK Lemamatizer

First I use nltk.pos_tag() to identify the part of speech for each word. This way I can loop through each word when applying NLTK Lemmatizer. This is as well the method I would use for large Text Datasets in order to identify part of speech. 

``` python
# create word list
words = ['better','ran', 'are', 'running', 'were', 'shared', 'organize', 'university', 
         'awoken', 'arose', 'beheld', 'sped', 'withhold', 'flung', 'cats', 'timely', 'actively', 'tighter', 'smaller', 'farther', 'driest', 'farthest', 'loudly']
         
#identify part of speech from word list
tags = nltk.pos_tag(words)

# create a list from part of speech
tag = list(dict(tags).values())

# rename part of speech for parameter input in NLTK Lemmatizer
for i in range(0, len(tag)): 
    if tag[i] == 'JJR' or tag[i] == 'JJ' or tag[i] == 'JJS':
        tag[i] = 'a'
    elif tag[i] == 'VBP' or tag[i] == 'VBG' or tag[i] == 'VBD' or tag[i] == 'VBN':
        tag[i] = 'v'
    elif tag[i] == 'NN' or tag[i]== 'NNS':
        tag[i] = 'n'
    elif tag[i] == 'RB':
        tag[i] = 'r'
    else:
        pass

# Create a new list with the word list and Part of speech    
word_list = []
for i in range(0, len(tag)):
    word_list.append([words[i], tag[i]])

# Loop to apply NLTK Lemmatizer
nltk_words = []
for word in word_list:
    print(word[0],'-->', lem.lemmatize(word[0], pos = word[1]))
    nltk_words.append(lem.lemmatize(word[0], pos = word[1]))
```
**Output**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; better --> good <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ran --> ran <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; are --> be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; running --> run <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; were --> be <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shared --> share <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; organize --> organize <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; university --> university <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; awoken --> awake <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; arose --> arose <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; beheld --> beheld <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; sped --> speed <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; withhold --> withhold <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; flung --> flung <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cats --> cat <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; timely --> timely <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; actively --> actively <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tighter --> tight <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; smaller --> small <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farther --> farther <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; driest --> driest <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; farthest --> farthest <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; loudly --> loudly <br>

### Comparison
Spacy Lemmatization is able to identify the root word for verbs, plural, adjectives, and for most adverbs. Spacy Lemmatization is not able to handle complex verbs as well as NLTK Lemmatizer when the POS is labelled. Such words includes 'flung', 'withold', and 'beheld'. One thing to note is that when I used NLTK.pos_tag() to identify the part of speech for each word in order to include the part of speech for the word list in NLTK Lemmatizer, not all of the part of speech were identified correctly. This is why when I applied NLTK Lemmatizer driest did not change to dry and flung do not change to fling.  

### Conclusion

After comparing NLTK Stemming, NLTK Lemmatizer, and Spacy Lemmatizer. Spacy Lemmatizer works the best out of the three. Note, for further research there are other Lemmatization packages that can be compared as well. However, I am happy how Spacy Lemmatizer works and for now, I will use this Lemmatization package for Text Processing.

#### References
[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"} 

Thomas, Rachel. “Topic Modeling with SVD & NMF (NLP Video 2).” YouTube, 8 July 2019, [www.youtube.com/watch?v=tG3pUwmGjsc](www.youtube.com/watch?v=tG3pUwmGjsc){:target="_blank"}. 

[https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"}

Porter, Martin F. 1980. An algorithm for suffix stripping. Program 14 (3): 130-137.

[http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer){:target="_blank"}

[http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize){:target="_blank"}

