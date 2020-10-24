---
layout: post
title: Collecting Twitter Data on the US 2020 Presidential Election
---
![Alternate image text](/images/twitter/tweet_tweet.jpg)

## Part 1: Data Collection
As a Data Scientist, it is easy to want to start with nicely formatted data and to jump into the predictive modelling and analysis phase. However, beginning a Data Science project from raw data can be fun and rewarding! Everyone loves a good challenge. Right? I decided for my raw data challenge, I would work with Twitter API data. By the way, can #RawDataChallenge become a thing? Anyways as I was saying, I decided to work with Twitter API data because 1) The documentation on Twitter APIs is abundant. Huzzah! 2) Twitter APIs provide very useful text data such as the tweets, location, hashtags, and user mentions, and other data includes time, date, and number of followers. 4) I am intrigued by the idea of finding twitter trends and identifying emotions and topics from just a simple tweet. 

### Purpose
I am collecting Twitter API data in order to analyze topics during the 2020 presidential election and to determine if there is a trend in the overall sentiment within these topics depending on location and time of the tweet. Topics that I expect to find from the tweets comprises of COVID19, Trump, Biden, election, vaccines, Black Lives Matter, fake news, Proud Boys, voter fraud, climate change, Ruth Ginsburg, Amy Coney Barrett, and TikTok. Phew, can you believe all this has happened in just 4 months?! 

![Alternate image text](/images/twitter/vote_facemask.jpg)


### The Data
The dataset I am using is originally taken from <a href="https://github.com/echen102/us-pres-elections-2020" target="_blank">GitHub</a> repository. Ok I know what you are thinking. “Wait! Aviel, I thought you were starting from scratch!” Hear me out first, and then you will see where I am going with this. The GitHub repository includes a list of ongoing tweet IDs associated with the 2020 United States presidential election from June 20, 2020 – October 2020. This means the repository will continue to be updated through November 3, 2020 and possibly beyond (Chen et. al 2020). This also means that I will need to obtain the json file associated with each Tweet ID. 

#### Step 1: How do I even collect twitter data?

I first applied for a [twitter developer](https://developer.twitter.com/en/apply-for-access){:target="_blank"} account. Four hours after applying, I was approved! Not going to lie, I felt pretty cool logging into my Twitter Developer Account. 

Next! I located my Developer Portal Dashboard. I created a Project App which provides me with a Key and Token. Keys are unique identifiers that authenticates a developer’s identity, and tokens are a type of authorization for a project app to gain access to the data.

There are multiple ways to collect tweet data and I have only scratched the surface. However, I will share with you my journey so far. 

Before collecting the tweets, I first download the data folders from the GitHub repo by applying a [fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo){:target="_blank"} to my personal GitHub repository. I then pulled the dataset from my forked repo to my desktop by running this git command: 

```
git clone https://github.com/user_name/repo_name.git

```

This step is important because the dataset will be continuously updated in the next month or two and I will want to easily update my folder without re-downloading the entire dataset every time.

Ok. I have the data on my desktop! Yay! Well sort of….

As I mentioned earlier, the dataset contains a list of Tweet IDs. In order to search the tweet API associated with the Tweet ID, I need to rehydrate the Tweet IDs with [Twarc](https://github.com/DocNow/twarc){:target="_blank"} or another installation package of your choosing. The reason for rehydrated tweets is because of Twitter API’s Terms of Service which does not permit developers to provide large amounts of raw Twitter data to be available on the Web. The goal of this is for Twitter to ensure user privacy. For example, if a user decides to delete their tweet, then the developer will not be able to retrieve that deleted tweet even with the tweet ID. As well part of the developer agreement, is that private information found in tweet data such as a username or user ID associated to sexual orientation, religion, health, or alleged crime is not permitted to made public on the web. Well that seems fair enough. 

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

Now that I have twark configured, I can run a command to retrieve tweet data. 

For example: 
```
twarc search #blacklivesmatter > tweets.jsonl
```
This will search for all tweets that contain the hashtag #blacklivesmatter and will be placed into a file named tweets.jsonl

The us-pres-elections-2020 repo from GitHub user [echen102](https://github.com/echen102){:target="_blank"} provides a python script ‘hydrate.py’ which runs twarc and rehydrates every tweet ID from the text files and creates a zipped json file with about 100,000 tweet data in each zipped json file. 

Run in the command line: 

    python3 hydrate.py 

<br>
![Alternate image text](/images/twitter/hydrate.png)

#### Step 2: Running into problems

After happily running the hydrate.py script on my computer for about four days, I realized that only June 2020 tweets had been extracted! I decided take a look at the us-pres-elections-2020 repo and review the associated [paper](https://arxiv.org/pdf/2010.00600.pdf){:target="_blank"}. In the paper it states that the first release of tweets from 6/20/2020 through 9/06/2020 contains **240 million tweets** which is almost **2 TB** of raw data! Within Twarc, I am only able to obtain 1 million tweets per day (this is part of the Twitter API limit) which means it would take 140 days to retrieve all that data!

**Ummmm... Abandon twarcing! Stop Collecting Data!!**

My MacBook Air cannot handle 240 million tweets and its storage limit is 250 GB lol. I could potentially collect the data on the cloud, however at the moment I do not want to work on the cloud or at least until my analysis is ready for production.

#### Step 3: Revisiting collecting twitter data

![Alternate image text](/images/twitter/rethinking.jpg)

I want to collect enough data for each month so that I can analyze the tweet data throughout the presidential campaign. Within the Twarc documentation they describe applying certain filters such as: 

```
twarc search blacklivesmatter \
  --30day docnowdev \
  --from_date 2020-05-01 \
  --to_date 2020-05-14 \
  --limit 1000 \
  > tweets.jsonl
```

However, with this project I already have the tweet ID. As well the tweet IDs are setup into folders: 2020-06, 2020-07, 2020-08, 2020-09, 2020-10. Each folder contains 

#### References

Emily Chen, Ashok Deb, Emilio Ferrara. #Election2020: The First Public Twitter Dataset on the 2020 US Presidential Election. Arxiv (2020)
