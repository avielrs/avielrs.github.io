---
layout: post
title: PART 1: Collecting Twitter Data on the US 2020 Presidential Election
---
![Alternate image text](/images/twitter/tweet_tweet.jpg)

**Collecting Twitter Data on the US 2020 Presidential Election**
In the beginning there was data (pretend that 2001: A Space Odyssey theme song playing in the background). This data was unstructured, floating in the internet, and not able to process. We took this data and made it readable for computers to process and formatted for humans to analyze. Ok I’ll get to the point.
As a Data Scientist, it is easy to get caught up with wanting to start with nicely formatted data and jumping into the predictive modelling and analysis phase. However, beginning a Data Science project from raw data can be fun and rewarding. Everyone loves a good challenge. Right? I decided it was time to work with Twitter APIs to create my own dataset. The reason for working with Twitter APIs is because 1) There is an insane amount of documentation on Twitter APIs. Score! 2) There is an overwhelming amount of data that can be collected from Twitter (which I will discuss in a later post), 3) Twitter APIs allows me to obtain information like the text, time/date, location of the tweet, number of followers in the user, hashtags, user mentions, and 4) I am intrigued by the idea of being able to create a story, a timeline, and find a trend in emotions on a specific topic just from a simple tweet.

### Purpose
I am collecting [Twitter API](https://developer.twitter.com/en/docs) data in order to analyze topics during the 2020 presidential election and to determine if there is a trend in the overall sentiment within these topics depending on state and time such as in June versus October. Topics that I expect to find from the tweets comprises of COVID19, Trump, Biden, election, vaccines, Black Lives Matter, fake news, Proud Boys, voter fraud, climate change, Ruth Ginsburg, Amy Coney Barrett, and TikTok. Phew, can you believe all this has happened in just 4 months?! 

### The Data
The dataset I am using is originally taken from [GitHub](https://github.com/echen102/us-pres-elections-2020)repository. Ok I know what you are thinking. “Wait! Aviel, I thought you were starting from scratch!” Ok, hear me out first and then you will see where I am going with this. The GitHub repository includes a list of ongoing tweet IDs associated with the 2020 United States presidential elections from June 20, 2020 – October 2020 which means the repository will continue to be updated through November 3, 2020 and possibly beyond (Chen et. al 2020). This also means that I will need to obtain the json file associated with each Tweet ID. 

When extracting a Twitter API, the data is encoded using [JSON]https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/intro-to-tweet-json). The data is then setup as a data dictionary where each object is given an attribute. 

#### Step 1: How do I even grab twitter data? And which data to collect?

I first applied for a [twitter developer](https://developer.twitter.com/en/apply-for-access) account and four hours later I was approved! Not going to lie, I felt pretty cool logging into my Twitter Developer Account. 

Next! I located my Developer Portal Dashboard. I created a Project App which provides me with a [Key and Token](https://developer.twitter.com/en/docs/authentication/oauth-2-0). Keys are unique identifiers that authenticates a developer’s identity, and tokens are a type of authorization for a project app to gain access to the data.

There are multiple ways to collect tweet data and I have only scratched the surface when it comes to collecting APIs. However, I will share with you my journey so far. 

Before collecting the tweets, I first download the data folders from the GitHub rep by applying a [fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) to my personal GitHub repository. I then pulled the dataset from my forked repo to my desktop by running this git command: 

```
git pull origin

```

This step is important because the dataset will be continuously updated in the next month or two and I want to easily be able to update my folder without re-downloading the entire dataset every time.

Ok. I have the data on my desktop! Yay! Well sort of….

As I mentioned earlier, the dataset contains a list of Tweet IDs. In order to search the tweet API associated with the Tweet ID, I need to rehydrate the Tweet IDs with [Twarc](https://github.com/DocNow/twarc) or another installation package of your choosing. The reason for rehydrated tweets is because of Twitter API’s Terms of Service which does not permit developers to provide large amounts of raw Twitter data available on the Web. The goal of this is for Twitter to ensure user privacy. For example, if a user decides to delete their tweet, then the developer will not be able to retrieve the deleted tweet even with the tweet ID. As well part of the developer agreement, is that private information found in tweet data such as a username or user ID associated to sexual orientation, religion, health, or alleged crime is not permitted to made public on the web. Well that seems fair enough. 

Install Twarc
```
pip install twarc
```

To begin using twarc enter in the command line:
```
twarc configure
```

Now I am ready to Twarc!! Do you think Ed Summers(edsu) and Github user SamSamhuns who developed twarc is making a pun from twerk? I may have to email them to find out.

Now that I have twark configured, I can run a simple command to retrieve tweet data. For example: 
```
twarc search #blacklivesmatter > tweets.jsonl
```
Which will search for all tweets with the hashtag #blacklivesmatter and will be placed into a file named tweets.jsonl

The us-pres-elections-2020 repo provides a python script ‘hydrate.py’ which will run twarc and search through each .txt file which lists about 100,000 tweet IDs for each file and outputs a zipped json file containing all twitter data associated with every tweet ID in that .txt file. 

#### Step 2: Running into problems

After happily running the hydrate.py script on my computer for four days, I realized that I had not even finished extracting tweets from June! So I went back to the us-pres-elections-2020 repo and reviewed the associated (paper)[https://arxiv.org/pdf/2010.00600.pdf]. In the paper it states their first release of tweets from 6/20/2020 through 9/06/2020 contains 240 million tweets which is almost 2 TB of raw data!

uh......ABANDON hydrate.py! STOP COLLECTING DATA!

My macbook air cannot handle 240 million tweets and its storage limit is 250 GB lol. I could potentially collect the data on the cloud, however at the moment I do not want to work on the cloud or at least until my analysis is ready to run.

This put me in a perdicament. I want to collect enough data for each month so that I can analyze the tweet data throughout the presedential campaign, rather than for just the month of June 2020. Within the Twarc documentation they describe applying certain filters such as: 


```
twarc search blacklivesmatter \
  --30day docnowdev \
  --from_date 2020-05-01 \
  --to_date 2020-05-14 \
  --limit 1000 \
  > tweets.jsonl
```

However, with how the data is setup in that 
#### References

Emily Chen, Ashok Deb, Emilio Ferrara. #Election2020: The First Public Twitter Dataset on the 2020 US Presidential Election. Arxiv (2020)
