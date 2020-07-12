### Step 1: Transform dataframe into a list

This dataset is a simple grocery store dataset taken from [Kaggle](https://www.kaggle.com/shazadudwadia/supermarket). The dataset is a list of 20 transactions from a grocery store. 



```python

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

![Alternate image text](/images/Intro_MBA/basket.png)

### Step 2: Transform input dataset into a one-hot encoded NumPy boolean array
```python
# Instantiate
te = TransactionEncoder()

# Fit and Transform the data into True and False (1 and 0)
item = te.fit(basket).transform(basket)

# Create DataFrame
df = pd.DataFrame(item, columns = te.columns_)
```

![Alternate image text](/images/Intro_MBA/basket_dummy.png)

Just by looking at this contingency heatmap which shows the frequency of each item purchased with another item. We can see some trends. For example, Bread is bought frequently with tea, sugar, milk, maggi, and coffee. As well, cornflakes and coffee were bought three times together. While for example, sugar and jam were never purachased together. While this is a first order attempt to look at the relationships, the Aipori Algorithm can provide an even more detail outline of relationships between multiple items which can provide much greater insight.


![Alternate image text](/images/Intro_MBA/basket_correlation.png)

### Step 3: Aipori Algorithm

```python
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True, max_len = 4)

# Add a column to the DataFrame that includes the length of the itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# In this specific project, we only care about itemsets where length = 2
# and support is greater than and equal to 0.05 (5%)
items = frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.05) ]

#take a look at the help for ways we can use this function
association_rules = association_rules(x, metric="lift", min_threshold=1)
```