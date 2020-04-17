---
layout: post
title: Use of NLP and Supervised learning to Target Wine Scores
---

#### Problem Statement:
Predicting how a wine will perform and understanding why that wine performs can be beneficial for consumers and sellers of wine. For example, a winemaker might want to gain insights into why their pinot noir is rated higher than their chardonnay. As well, wine distributers might want to make profitable decisions like is it worth it to invest in an expensive wine and will this wine sell. To delve into these questions, I created a predictive model to target wine scores and analyze a dataset that includes the geography of the wine, price, description, variety, and vintage.

#### Background:
Understanding what makes a great wine is very complex. The taste of a wine is dependent on the climate, weather, geology, soil, timing of pruning, and chemical makeup of the wine. For example, a pinot noir thrives in Central Coast, California because that region provides the combination of sunlight and fog which creates a cool climate that is perfect for winemaking. As well the geology in that region provides soil that contains mudstone and siltstone which is another contributing factor to a great tasting pinot noir. We can see that creating a predictive model to determine which wines perform the best is complex. Therefore, this project is a first step attempt to predicting wine scores and perfecting this predictive model entails a multi-iterative process.

#### Part 1: Cleaning Data


Step 1: Import Packages

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Next import csv dataframe `winemag_data_130k_v2.csv`

Step 2: Split train and test data

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

