---
layout: post
title: Use of NLP and Supervised learning to Target Wine Scores
---
![Alternate image text](/images/NLP_Wine/wine1.jpg)

#### Problem Statement:
Predicting how a wine will perform and understanding why that wine performs well can be beneficial for consumers and sellers of wine. For example, a winemaker might want to gain insights into why their pinot noir is rated higher than their chardonnay. As well, wine distributers might want to make profitable decisions like is it worth it to invest in an expensive wine and will this wine sell. To delve into these questions, I created a predictive model to attempt to target wine scores and analyze a dataset that includes the geography of the wine, price, description, variety, and vintage.

#### Background:
Understanding what makes a great wine is very complex. The taste of a wine is dependent on the climate, weather, geology, soil, timing of pruning, and chemical makeup of the wine. For example, a pinot noir thrives in Central Coast, California because that region provides the combination of sunlight and fog which creates a cool climate that is perfect for winemaking. As well the geology in that region provides soil content of mudstone and siltstone which is another contributing factor to a great tasting pinot noir. We can see that creating a predictive model to determine which wines perform the best is complex. Therefore, this project is a first step approach to predicting wine scores.

#### Setup:
The goal of this project is predict wine scores from wine reviews, geography, variety, and vintage. I downloaded a wine dataset from [Kaggle](https://www.kaggle.com/zynicide/wine-reviews) which will be used to predict the wine scores. Because I am predicting a target value from multiple independent variables this will be a Supervised Learning model.

--- 

### Part 1: Data Cleaning and Data Wrangling

#### Step 1: EDA:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

#### Define each column in the [wine reviews dataset](https://www.kaggle.com/zynicide/wine-reviews): 
- country: Name of the Country where the wine is grown
- description: Description of the wine review
- designation: The vineyard within the winery where the grapes that made the wine
- points: Rating of the wine ranging 80 - 100
- price: Price of a bottle of wine
- province: Name of the Province/State where the wine is grown
- region_1: Region within the province where the wine is grown
- reigon_2: A sub-region within the region where the wine is grown
- taster_name: Name of the person who rated the wine
- taster_twitter_handle: The twitter handle of the person who rated the wine
- tile: Name of the wine
- variety: The blend/type of wine
- winery: Name of the winery where the wine is made.


#### Things to note: 

- points coloumn is our target and will be set as the dependent variable Y
- taster_twiter_handle will be deleted as it will not be included in the model
- Each row of data is a different review
- Each column describes information about the wine, review, and reviewer
- I can extract the Vintage (year) of the wine from the title column
- NaN values are present
- taster_twiter_handle will be deleted as it will not be included in the model
- region_2 will be removed from the dataset because text data will really enlarge the size of the dataset, as well region_2 seems to sometimes replicate the same value as in region_1 (see row 2 from above). Having this similarity can cause a multi-colinearity problem in the model.
- I have also decided to drop region_1 because the dataset will become very large once I use NLP on the text columns. I can always include region_1 later on to see how the models run.
- I will drop designation for the same reason for dropping region_1

#### Step 2: Split train/test set

The train dataset will be used to train the model and the test dataset will be used to test how well the model runs on a dataset that is not trained. I split the datatset before even cleaning so that I can make sure all steps can be reproduced for a future dataset. As well, I will not be "peaking" into the test dataset when making decisions on how to clean the data and how to model the data.

```python
# Import train_test_split package
from sklearn.model_selection import train_test_split

# Split data into train and test, where text_size is 30 percent, andsp train set is 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify = y)
```

Step 3: Explore Data
Before removing missing data and dropping columns, it is important to first review the dataset. 
    
Things to check in a dataset:

1. Make sure each column contains the correct data type
2. Check to see if a column name should be renamed
3. Check to see if there are duplicate columns
4. How many rows in each column
5. Are there missing values in a column? 
6. Can we complete feature engineering from a column

#### Step 4. Drop Columns
I dropped all columns that seemed to be less helpful for the model. This is a qualitative decision. For a more quantitative decision, I could look at the variance for each column. If there is high variance than I might want to consider keeping that column, while a column with low variance, I might want to consider dropping.

```python
# Drop columns
X_train = X_train.drop(['designation', 'taster_twitter_handle', 'winery', 'taster_name', 'region_2', 'region_1'], axis=1)
```

#### Step 5. Fill Missing Values

```python
# Replace variety nan with most frequent variety
X_train['variety'].fillna(X_train['variety'].describe()['top'], inplace = True)
```

---

### Part 2: Feature Engineering
Goal: Extract the year of the wine from the title column

Import regex package:
``` python
import re
```

Use regex to find and extract the year from the title column.
``` python
# Define regex to find all years that range between 1900 - 2019 
regex = '([1][9][0-9][0-9]|[2][0-1][0-2][0-9])'     
``` 

To view entire code for extracting year from title please visit my [github](https://github.com/avielrs/BrainStation-Capstone/blob/master/Notebook/Part1_Clean.ipynb)

---

### Step 3: View Classification of wine score

![Alternate image text](/images/NLP_Wine/winescoredistribution.png)

The wine scores ranges between 80 – 100 with an increment of 1. There are three approaches for creating a predictive model to target these wine scores. One approach is to use a regression model and treat the wine scores as a continuous quantity. The second approach is to use a classification model to predict the wine scores and treat the wine scores as 20 classifiers. While technically we can have 20 classifiers in a classification model, this would not be helpful for predicting a score because the accuracy score will be too low in a classification model. In order to run a classification model, I group the wine scores into two categories by setting a score of 90 and above to equal 1 and a score below 90 to equal 0. By doing this we have two classifications where we predict “good” scores and “bad” scores.  We can see from the figure that when I group the scores into 0 and 1, there are more negative scores than positive scores therefore the predictive model will be better at predicting wine scores that negative. 

![Alternate image text](/images/NLP_Wine/true_false_distribution.png)


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



<iframe src="https://public.tableau.com/views/test_15948342860330/WeightedWords?:showVizHome=no&:embed=true"></iframe>



```python
# View tokens associated to their weight for the train dataset
word_weights = np.array(np.sum(review_train, axis=0)).reshape((-1,))

words = np.array(tfidf.get_feature_names())
words_df = pd.DataFrame({"word": words, 
                         "weight": word_weights}) 
```

 
