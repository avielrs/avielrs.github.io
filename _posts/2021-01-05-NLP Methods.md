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

**Definition of Natural Language Processing**
Natural Language Processing is part of the machine learning/AI pipeline, where a variety of tasks are applied in order to process text data and format it in a way so that the computer can read the data and perform analysis. 

*In this blog post I will discuss a few different approaches for pre-processing text so that the text is ready to process for machine learning and rule-based algorithms. Specifically, in regards to topic modelling with the use of Twitter text data. This blog post is Part 3 in a series of posts in regard to [collecting twitter data on the US Presidential Election](https://avielrs.github.io/Collecting-Twitter-Data-on-the-US-Presidential-Election/){:target="_blank"}.*

## Stemming and Lemmatization
![Alternate image text](/images/twitter/dictionary.jpg)

Each document contains a vector of words (terms), in this case, the document is the tweet. Sentence tokenization separates each word into a matrix where each term is a feature. For example, if a sentance (or document) contains the term **sit**, and another document contains the term **sitting**. The terms will end up in separate columns even though the meaning is the same. This is where stemming and lemmatization may be beneficial. Stemming and Lemmatization are two separate approaches for stripping a term within a document so that a document matrix is reduced and thus the complexity of data decreases. Reducing size and complexity of a model is beneficial for achieving model accuracy and for reducing computationally memory and time.

Two approaches to reducing the term: Stemming and Lemmatization. 

### Stemming

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

![Alternate image text](/images/twitter/stem.png)

Pros: 
1. Remove sufixes
2. Reduce size and complexity of data
3. Reduces variance in models which can cause overfitting.

Cons: Stemming does "too" good of a job of extracting the root word. What I mean by this is that the stemming function will take a word like organize and shorten it to organ which has an entirely different meaning. Same with the word University which will stem to univers, again shortening the word to a word that is not the same meaning. This is something to consider when using Porter Stemmer. How important is the meaning of the word versus reducing the complexity of the data to your model and analysis?

**Lemmatization** is another approach that handles terms by labeling the term from its base word (lemma). With this method, it is ensuring that you are not grouping terms together with different means like universe and university done in stemming.  

The base word is Recall conjugating verbs from Spanish or French class? Let's start at the beginning. What is the conjugation of 'To Be'? 

Present Tense: <br>
I am <br>
You are  <br>   
He/She/It is <br><br>
We are <br>
You are <br>
They are <br>

Past Tense: <br>
I was   <br> 
You were   <br> 
He/She/It was <br><br>
We were <br>
You were <br>
They were <br>

The lemma or lemmatization of AM, IS, ARE, WAS, and were is 'BE'

We can test this out in code: 

![Alternate image text](/images/twitter/to_be_lem.png)

**Note that I needed to define the part of speech as a verb. If part of speech parameter is not defined then the default is set to NOUN.**

![Alternate image text](/images/twitter/lem.png)

Even though part of speech is not identified, we can see here that Lemmatization is more conservative about trimming a word. University does not change to universe and organize/organization does not change to organ.

If part of speech is applied, Lemmatization is great at even identify9ng the base word for complex verbs. For example

![Alternate image text](/images/twitter/lem_complex_verbs.png)

Do not need to identify part of speech for plural: 
![Alternate image text](/images/twitter/lem_plural.png)

Identifying part of speech for nouns and adjectives is more meaningful within languages other than English. Basically, languages that utilize grammatical gender. 

For example, the Hebrew word for 'big' is גָּדוֹל (gadol):

![Alternate image text](/images/twitter/gadol.png)

Here the adjective changes the ending of the word depending if the adjective is describing a single masculine, single feminine, plural masculine, or plural feminine. However, the root (lemma) of the word of Gadol is  ג - ד - ל which is the simplest form of the word because the vowels and gender are removed. 

Pros: 
1. Using the base word ensures that the meaning behind the word is not being lost

Cons: 
1. Need to identify part of speech
2. Need to understand fundamentals of linguistics thus more complex
3. Not as good as stemmer for query use 

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
