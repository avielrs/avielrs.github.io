---
layout: post
title: Collecting Twitter Data on the US 2020 Presidential Election
---
![Alternate image text](/images/twitter/tweet_tweet.jpg)

# Part 1: Data Collection
As a Data Scientist, it is easy to want to start with nicely formatted data and to jump into the predictive modelling and analysis phase. However, beginning a data science project from raw data can be fun and rewarding! Everyone loves a good challenge. Right? I decided for my raw data challenge, I will work with Twitter API data. By the way, can #RawDataChallenge become a thing? Anyways as I was saying, I decided to work with Twitter API data because 1) The documentation on Twitter APIs is abundant. Huzzah! 2) Twitter APIs provide very useful text data such as tweets, location, hashtags, and user mentions. Other data includes time, date, and number of followers. 3) I am intrigued by the idea of finding twitter trends and identifying major topics from just a simple tweet. 

### Purpose
I am collecting Twitter API data in order to analyze topics during the 2020 presidential election and to determine if there is a trend in the overall sentiment within these topics depending on location and time of the tweet. Topics that I expect to find from the tweets comprises of COVID19, Trump, Biden, election, vaccines, Black Lives Matter, fake news, Proud Boys, voter fraud, climate change, Ruth Ginsburg, Amy Coney Barrett, and TikTok. Phew, can you believe all this has happened in just 4 months?! 

![Alternate image text](/images/twitter/vote_facemask.jpg)

### The Data
The data I am collecting is based off of a <a href="https://github.com/echen102/us-pres-elections-2020" target="_blank">GitHub repository</a>. The GitHub repository includes a list of ongoing tweet IDs associated with the 2020 United States presidential election from June 20, 2020 – October 2020. This means the repository will continue to be updated through November 3, 2020 and possibly beyond (Chen et. al 2020). This also means that I will need to obtain the json file associated with each Tweet ID. 

#### Step 1: How do I even collect twitter data?

First, I applied for a [twitter developer](https://developer.twitter.com/en/apply-for-access){:target="_blank"} account. Four hours later, I was approved! Not going to lie, I felt pretty cool logging into my Twitter Developer Account. 

Next! I located my Developer Portal Dashboard. I created a Project App which provides me with a Key and Token. Keys are unique identifiers that authenticates a developer’s identity, and tokens are a type of authorization for a project app to gain access to the data.

There are multiple ways to collect tweet data and I have only scratched the surface. However, I will share with you my journey so far. 

Before collecting the tweets, I first downloaded the data folders from the GitHub repo by applying a [fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo){:target="_blank"} to my personal GitHub repository. I then pulled the dataset from my forked repo by running this git command: 

```
git clone https://github.com/user_name/repo_name.git
```

This step basically downloads the most recently updated files from the repo onto the computer. This step is important because the dataset will continuously update in the next month or two. Cloning the git folder will allow me to easily update my files without re-downloading the entire dataset every time.

Ok. I have the data on my desktop! Yay! Well sort of….

As I mentioned earlier, the dataset contains a list of tweet IDs. The tweet IDs were obtained by GitHub user [echen102](https://github.com/echen102){:target="_blank"} and the data science research group at USC. They were obtained by hitting the twitter API endpoints. In order to search for tweets regarding the 2020 Presidential election, filters were added to search for specific tweet accounts and tweets with specific users mentioned. After collecting the twitter APIs, a provided and organized list of the tweet IDS associated to these tweets were posted on GitHub. 

The reason why a list of tweet IDs are provided, rather than the entire json file is because of Twitter API’s [Terms of Service agreement](https://developer.twitter.com/en/developer-terms/agreement-and-policy){:target="_blank"}. The agreement states that developers cannot make large amounts of raw Twitter data available onto the Web. This is to ensure user privacy on Twitter. For example, if a user decides to delete their tweet, then the developer will not be able to retrieve that tweet. As well, private information found in tweet data such as a username or user ID associated to sexual orientation, religion, health, or alleged crime is not permitted to make available to the public on the web. Well, that seems fair enough. 

In order to utilize the tweet IDs, I need to **rehydrate**. To rehydrate tweet ids means to take a tweet ID and extract the entire tweet API information associated to that ID. 

#### Part 3: Rehydrate 

I decide to rehydrate with the package [Twarc](https://github.com/DocNow/twarc). Another package out there is [Hydrator](https://github.com/DocNow/hydrator) (GUI version). I could as well directly hit the twitter API endpoint to retrieve the APIs. 

Twarc is a command line tool that archives twitter JSON data. What makes Twarc beneficial is that Twarc handles the twitter rate limit for the user and thus able to handle extracting 1 million tweet IDs per day! Whereas, Twitter API v2 rate limit is 900 requests/15-minutes thus 43,200 requests per day. The total amount of requests per month is limited to 500,000 per user! The Twitter's rate limit helps manage large volumes of requests which are placed by thousands of developers each day on twitter. 

#### Step 2: Install packages

Install Twarc:
```
pip install twarc
```

To begin using twarc enter in the command line:
```
twarc configure
```

The below image should appear in your Terminal: Happing Twarcing! <br>
Do you think the developers of twarc are making a pun from Twerking? I may have to email them and ask.<br><br>
![Alternate image text](/images/twitter/happy_twarcing.png)

Now that I have configured twarc, I can run a command to retrieve tweet data. 

For example: 
```
twarc search #blacklivesmatter > tweets.jsonl
```
This will search for all tweets that contain the hashtag #blacklivesmatter and will be placed into a file named tweets.jsonl

The us-pres-elections-2020 repo from GitHub user [echen102](https://github.com/echen102){:target="_blank"} provides a python script ‘hydrate.py’ which runs twarc and rehydrates every tweet ID from the text file. The tweet IDs from each file are zipped into a json file which contains about 100,000 tweet data. 

Run in the command line: 

    python3 hydrate.py 

<br>
![Alternate image text](/images/twitter/hydrate.png)

#### Step 3: Running into problems

After happily running the hydrate.py script on my computer for about four days, I realized that extracting tweets from June 2020 is not even complete! I decided to take a look at the us-pres-elections-2020 repo and review the associated [paper](https://arxiv.org/pdf/2010.00600.pdf){:target="_blank"}. In the paper it states that the first release of tweets from 6/20/2020 through 9/06/2020 contains **240 million tweets** which is almost **2 TB** of raw data! 

Anyways back to 240 million tweets and 2 TB of data: <br>
**Abandon twarcing! Stop Collecting Data!!**

My MacBook Air cannot handle 240 million tweets and its storage limit is 250 GB lol. I could potentially collect the data on the cloud, however at the moment I do not want to work on the cloud or at least until my analysis is ready for production.

#### Step 3: Revisiting collecting twitter data

![Alternate image text](/images/twitter/rethinking.jpg)

I want to collect enough data for each month so that I can analyze the tweet data throughout the presidential campaign. Within the Twarc documentation they describe applying certain filters such as a date range and limiting number of tweets:

```
twarc search blacklivesmatter \
  --30day docnowdev \
  --from_date 2020-05-01 \
  --to_date 2020-05-14 \
  --limit 1000 \
  > tweets.jsonl
```

However, with this specific project I already have an extensive list of tweet IDs regarding the 2020 election. The tweet IDs are setup into folders: 2020-06, 2020-07, 2020-08, 2020-09, 2020-10. Each folder contains multiple .txt file and each .txt file contains about 100,000 tweet IDs <br>

<p align="center">
    <img src="/images/twitter/text_file.png"/>
</p>

<br>
To hydrate tweet identifiers that are contained in a .txt file in twarc, run the command :  

    for tweet in t.hydrate(open('ids.txt')):
      print(tweet["text"])

This method does not include a filter method. I am currently still trying to figure out how to add a filter to this command. In the meantime, I decided to filter the old fashion way! I am deleting the .txt files myself. There are multiple .txt files for each day, therefore I only kept one .txt file for each day. To be honest it really didn't take long to delete the files manually. Recall that each .txt file contains about 100,000 tweets. This means I will have about 100,000 tweets regarding the US 2020 presidential election for every day between June 2020 - November 2020. This is still an overabundance of data for this specific project. 

### What's Next? 
#### Data Cleaning, Exploratory Data Analyst, Topic Labelling, Sentiment Analyst
![Alternate image text](/images/twitter/goals.jpg)
*Anyone else think the text in this picture said gouls instead of goals?* <br><br>
Overcome with excitement to apply sentiment analysis and topic modelling to the US Presidential 2020 twitter data, I almost forgot about data cleaning! Data cleaning is an essential step in the data science pipeline. Cleaning data is a preprocessing step that ensures your data is ready to be trained, tested, and analyzed. 

Here is a general guideline for my next steps on this project:

1.  Import the json files into python and restructure the data in tables and lists. 
2.  Data Cleaning and Data Wrangling
3.  Apply multiple Natural Language Processing techniques to determine the best approach.
4.  Apply sentiment analysis and topic modelling to find specific topics and trends: 
    - As whole within the US
    - As whole within each state
5. Did the sentiment of the topics swayed overtime?
6. Create a timeline of events from the tweets
<br>
<br>

#### To end this post, I will leave you with this quote.
*“‘Possessed’ is probably the right word. I often tell people, ‘I don’t want to necessarily be a data scientist. You just kind of are a data scientist. You just can’t help but look at that data set and go, ‘I feel like I need to look deeper. I feel like that’s not the right fit.’”* <br> 
― Jennifer Shin, Senior Principal Data Scientist at Nielsen; Lecturer at UC Berkeley
<br><br>
### References

Emily Chen, Ashok Deb, Emilio Ferrara. #Election2020: The First Public Twitter Dataset on the 2020 US Presidential Election. Arxiv (2020)
