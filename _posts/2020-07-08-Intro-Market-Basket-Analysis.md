---
layout: post
title: Introduction to Market Basket Analysis
---

Data taken from Kaggle. I picked this dataset because I wanted to gain experience applying Market Basket Analysis with a small dataset. Market Basket Analysis is a data mining technique and rule-based algorithm that can learn through relationships and can be utilized within ecommerce and marketing strategies. Through this analysis we can make smarter decisions that will benefit the consumer. For example, Market Basket Analysis helps provide insights that allows us to determine where to place items in a store. For example, we tend to see coffee, tea, and cereal placed in the same isle. As well, if a customer purchases one item we might consider targeting them with advertisements with another item. With MBA we can determine how to consider more advantageous group discounts, and even go one step further in developing a recommmendor system for ecommerce shoppers.

MBA is considered a type of rule based learning. A rule might look like: 'If a customer bought tortilla chips, then they will buy salsa.' This example might seem obvious, however we can create more specific association rules such as:

If a customer is female, aged 20-40, and buys diapers between 5 and 7pm, then customer will also buy wine.
Mining retail datasets like this is done to find a number of relations:
- Complementary products: products which are often bought together, like chips and salsa
- Substitute products: products which replace each other, like Coke and Pepsi
- Trigger products: products which when bought, trigger other purchases
- Common Baskets: combinations of products that are often bought together

#### Step 1
In order to run the market basket analysis, we must first start with a list of transactions: 
    
    Transaction 1: 'Apple', 'Beer', 'Rice', 'Chicken'
    Transaction 2: 'Apple', 'Beer', 'Rice' 
    Transaction 3: 'Apple', 'Beer'
    Transaction 4: 'Apple', 'Bananas'
    Transaction 5: 'Milk', 'Beer', 'Rice', 'Chicken'
    Transaction 6: 'Milk', 'Beer', 'Rice'
    Transaction 7: 'Milk', 'Beer'
    Transaction 8: 'Apple', 'Bananas'

    dataset = [['Apple', 'Beer', 'Rice', 'Chicken'], 
           ['Apple', 'Beer', 'Rice'], 
           ['Apple', 'Beer'], 
           ['Apple', 'Bananas'], 
           ['Milk', 'Beer', 'Rice', 'Chicken'], 
           ['Milk', 'Beer', 'Rice'], 
           ['Milk', 'Beer'], 
           ['Apple', 'Bananas']]
 
#### Step 2:
Next use [TransactionEncoder](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/) to transform the list of transactional items into dummy variables which is suitable for computating text data such as in Machine Learning and Statistical Analysis methods.

![Alternate image text](/images/dummytable.png)

#### Step 3:
Once the dataframe is setup correctly, we can run Aipori Algorithm which is an association rule algorithm. Association Rules "help uncover all such relationships between items from huge databases". The Aipori Algorithm groups the list of items into antecedents and consequents. The antecedent is what the customer purchased such as bread and eggs, while the consequent is the purchase result. For example, if a user purchases  beer (antecendent) then they will purcahse rice (consquent). We can see from the sample dataset above that if a customer buys beer (antecendent) then the customer buys rice (Consequent) for 50% of transactions (4 transacations/ 8 total transactions). We can see here that the antecedent and consequent are setup as an *if* (antecedent) *else* (consequent) statement.

Aipori Algorithm quantifies the likelihood of a customer who purchases item A who will also purchase item B.

Aipori Algorithm provides three components: Support, Confidence, Conviction, and Lift:

![Alternate image text](/images/support.png)

We can then set a support threshold where the support value means the item has a meaningful outcome on sales. Therefore identifying all items within all transactions where items contain a support threshold equal or greater than the set value.

Confidence signifies the likelihood of item Y being purchased with item X. This is also known as conidtional probablity P(Y|X). The conditional probability of P(Y|X) is the probability of itemset 𝑌 in all transactions given the transaction already contains 𝑋. The drawback of confidence is that it only takes into account the popularity of X, and not the popularity of Y.

![Alternate image text](/images/confidence.png)

Lift takes into account for popularity of Y which thus accomodates for the drawback present in calculating confidence. More precisely lift signifies the liklihood of item Y being purchased when item X is purchased, while taking into account the popularity of Y. If Lift > 1, then Y is likely bought with item X. Lift < 1, then Y is unlikely bought with item X.

![Alternate image text](/images/lift.png)

The lift of purchasing beer and rice together is 1.3 which means that the likelihood of a customer buying both beer and rice together is 1.3 times more than the chance of purchasing beer alone.<br/><br/>

Leverage is the difference in support of the larger group, than would be expected if the antecedent and consequent were independent: <br/>

![Alternate image text](/images/leverage.png)

Conviction is the measure of the dependence of the consequent on the antecedent: A high value denotes that we always purchase the C with the A. <br/>

![Alternate image text](/images/conviction.png)

#### Step 4: 
Create a recommendor system from lift value

### Step 1: Transform dataframe into a list

```python
# Import packages
import pandas as pd
import numpy as np
import matplotlib as pl

# Import mlxtend packages
from mlxtend.preprocessing import TransactionEncoder
import itertools  

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

```

```python
basket = []

df.values[0, 0].split(',')

for i in range(len(df)):
    basket.append(df.values[i, 0].split(','))
```

![Alternate image text](/images/basket.png)

### Step 2: Transform input dataset into a one-hot encoded NumPy boolean array
```python
# Instantiate
te = TransactionEncoder()

# Fit and Transform the data into True and False (1 and 0)
item = te.fit(basket).transform(basket)

# Create DataFrame
df = pd.DataFrame(item, columns = te.columns_)
```

![Alternate image text](/images/basket_dummy.png)

Just by looking at this contingency heatmap which shows the frequency of each item purchased with another item. We can see some trends. For example, Bread is bought frequently with tea, sugar, milk, maggi, and coffee. As well, cornflakes and coffee were bought three times together. While for example, sugar and jam were never purachased together. While this is a first order attempt to look at the relationships, the Aipori Algorithm can provide an even more detail outline of relationships between multiple items which can provide much greater insight.


![Alternate image text](/images/basket_correlation.png)

### Step 3: Aipori Algorithm

    min_support = is the frequency of occurance in the dataset
    use_colnames is our output
    max_len is the upper length 
    Support - what porportions of transactions include this set of items or include this basket.

    When max_len = 3, itemsets range between 1 item, 2 items, and 3 items
    min_suport - 0.1, minimum of 10% of  frequency of occurance

apriori(df, min_support=0.1, use_colnames=True, max_len = 3)

Links for Reference:
https://towardsdatascience.com/mba-for-breakfast-4c18164ef82b
https://www.youtube.com/watch?v=WGlMlS_Yydk&t=8s
https://towardsdatascience.com/association-rules-2-aa9a77241654
https://towardsdatascience.com/complete-guide-to-association-rules-2-2-c92072b56c84

