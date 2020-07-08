---
layout: post
title: Introduction to Market Basket Analysis
---

Data taken from Kaggle. I picked this dataset because I wanted to gain experience applying Market Basket Analysis with a small dataset. Market Basket Analysis is a data mining technique and rule-based algorithm that can learn through relationships and can be utilized within ecommerce and marketing strategies. Through this analysis we can make smarter decisions that will benefit the consumer. For example, Market Basket Analysis helps provide insights that allows us to determine where to place items in a store. For example, we tend to see coffee, tea, and cereal placed in the same isle. As well, if a customer purchases one item we might consider targeting them with advertisements with another item. With MBA we can determine how to consider more advantageous group discounts, and even go one step further in developing a recommmendor system for ecommerce shoppers.

MBA is considered a type of rule based learning. A rule might look like: 'If a customer bought tortilla chips, then they will buy salsa.' This example might seem obvious, however we can create more specific association rules such as:
If a customer is female, aged 20-40, and buys diapers between 5 and 7pm, then customer will also buy wine.
Mining retail datasets like this is done to find a number of relations:
•	Complementary products: products which are often bought together, like chips and salsa
•	Substitute products: products which replace each other, like Coke and Pepsi
•	Trigger products: products which when bought, trigger other purchases
•	Common Baskets: combinations of products that are often bought together

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

![Screen%20Shot%202020-06-17%20at%205.52.02%20PM.png](attachment:Screen%20Shot%202020-06-17%20at%205.52.02%20PM.png)

#### Step 3:
Once the dataframe is setup correctly, we can run Aipori Algorithm which is an association rule algorithm. Association Rules "help uncover all such relationships between items from huge databases". The Aipori Algorithm groups the list of items into antecedents and consequents. The antecedent is what the customer purchased such as bread and eggs, while the consequent is the purchase result (https://towardsdatascience.com/association-rules-2-aa9a77241654). For example, if a user purchases milk and sugar (antecendent) then they will purcahse coffee (consquent). We can see from the sample dataset above that if a customer buys beer (antecendent) then the customer buys rice (Consequent) for 50% of transactions (4 transacations/ 8 total transactions). We can see here that the antecedent and consequent are setup as an *if* (antecedent) *else* (consequent) statement.

![image.png](attachment:image.png)


While we might be able to determine through common behaviors that when a customer buys eggs and bread they will also buy milk. The Aipori Algorithm quantifies the likelihood of a customer who purchases eggs and bread who will also buy milk.

Aipori Algorithm provides three components: Support, Confidence, and Lift


$$  \text{support}(X \cup Y) = \frac{\text{# of transactions with X and Y together}}{\text{total number of transactions}} $$

#### Step 4: 
Create a recommendor system from lift value
Once the dataframe is setup correctly, we can run Aipori Algorithm which is an association rule algorithm. Association Rules "help uncover all such relationships between items from huge databases". The Aipori Algorithm groups the list of items into antecedents and consequents. The antecedent is what the customer purchased such as bread and eggs, while the consequent is the purchase result (https://towardsdatascience.com/association-rules-2-aa9a77241654). For example, if a user purchases milk and sugar (antecendent) then they will purcahse coffee (consquent). We can see from the sample dataset above that if a customer buys beer (antecendent) then the customer buys rice (Consequent) for 50% of transactions (4 transacations/ 8 total transactions). We can see here that the antecedent and consequent are setup as an if (antecedent) else (consequent) statement.

Links for Reference:
https://towardsdatascience.com/mba-for-breakfast-4c18164ef82b
https://www.youtube.com/watch?v=WGlMlS_Yydk&t=8s
https://towardsdatascience.com/association-rules-2-aa9a77241654
https://towardsdatascience.com/complete-guide-to-association-rules-2-2-c92072b56c84

