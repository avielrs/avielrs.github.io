---
layout: post
title: Part 2 Stemming vs Lemmatization
---
![Alternate image text](/images/twitter/books.jpg)
 
# II. Lemmatization
![Alternate image text](/images/twitter/dictionary.jpg)
 
**Lemmatization** labels the term from its base word (lemma). This method is a more methodical approach for ensuring word reduction does not lose its meaning.
 
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
To better understand how lemma applies within linguistics, let's take a trip down memory lane and recall verb conjugation.
 
**Conjugate 'To Be'**
 
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
 
Thus, the lemma or lemmatization of AM, IS, ARE, WAS, and WERE is BE.
 
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
 
**Note:** In NLTK lemmatization, the part of speech (pos) needs to be defined. In the example above, I define the pos as "v" for verb. If the pos parameter is not specified, then the default is set to NOUN.<br>
 
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
 
NLTK Lemmatizer can handle most adverbs, such as the words farther and loudly. However, adverbs that end in 'est', also known as superlative adverbs, is not supported by NLTK Lemmatizer. Therefore, loudest does not change to loud and farthest does not change to far.
 
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
 
### Pros for lemmatization
1. Using the base word ensures that the meaning behind the word is not lost
 
### Cons for lemmatization
1. Need to identify the part of speech
2. Need to understand the fundamentals of linguistics thus more complex
 
## B. Spacy Lemmatization
Spacy provides a lemmatization package. Let's see how this package compares with NLTK lemmatization.
 
### Spacy Lemmatization
In Spacy lemmatization, part of speech for each word is not a parameter.
 
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
 
First I use nltk.pos_tag() to identify the part of speech for each word. This way, I can loop through each word when applying NLTK Lemmatizer. This is as well the method I would use for large Text Datasets in order to identify part of speech.
 
``` python
# create word list
words = ['better','ran', 'are', 'running', 'were', 'shared', 'organize', 'university',
     	'awoken', 'arose', 'beheld', 'sped', 'withhold', 'flung', 'cats', 'timely', 'actively', 'tighter', 'smaller', 'farther', 'driest', 'farthest', 'loudly']
        
#identify part of speech from word list
tags = nltk.pos_tag(words)
 
# create a list from part of speech
tag = list(dict(tags).values())
 
# rename part of speech parameter input in NLTK Lemmatizer
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

Spacy Lemmatization can identify the root word for verbs, plural, adjectives, and most adverbs, but cannot handle complex verbs such as words 'flung', 'withhold', and 'beheld'. When setting the pos parameter to ‘v’, NLTK Lemmatizer changes flung to fling, however when I used NLTK.pos_tag() to identify part of speech, flung did not change. Other words that did not change because NLTK.pos_tag() did not correctly identify part of speech for driest, farther, arose, withhold, and beheld. NLTK.pos_tag() is necessary to use with NLTK Lemmatizer when working with thousands of documents. 
 
#### References

[B., Sowmya V., et al. Practical Natural Language Processing: a Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media, 2020.](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/){:target="_blank"}
 
Thomas, Rachel. “Topic Modeling with SVD & NMF (NLP Video 2).” YouTube, 8 July 2019, [www.youtube.com/watch?v=tG3pUwmGjsc](www.youtube.com/watch?v=tG3pUwmGjsc){:target="_blank"}.
 
[https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"}
 
Porter, Martin F. 1980. An algorithm for suffix stripping. Program 14 (3): 130-137.
 
[http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer){:target="_blank"}
 
[http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize){:target="_blank"}
 
 

