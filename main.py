
"""
Arav Popli & Pranay Nandkeolyar
Period 3 
Intermediate Data Programming 

This module runs a trading model for the cryptocurrency 
dogecoin. It attempts to profit off Elon Musk's tweets using 
a natural language processsing model to determine the sentimment of his 
tweets regarding dogecoin and trade off of it accordingly. It also creates 
a multitude of data visualizations in order to portray our findings. 
This required the use of many libraries including scipy (stats), quantstats (stats),
pandas (CSV processing), matplotlib (data visualizations), and nltk (NLP sentiment). 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import nltk
# Found out that vader_lexicon was required to be downloaded via ChatGPT
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import re


def main():
    """
    This method plots the graph and runs the training model 
    """
    # Unpack tuple of time series dataframes
    stock_data, tweet_data = get_data('DOGE-USD.csv', 'elonmusk.csv')
    plot_r_squared(stock_data, tweet_data)
    plot_volatility(stock_data, tweet_data)
    example_trade(stock_data, tweet_data)
    # we check the strategy of only buying when elon tweets
    # something positive about doge
    temp = add_allocation(stock_data, tweet_data, True)
    t_test_tweet_volatility(temp, tweet_data, True)
    trade_data_1 = calculate_profit(temp)
    plot_trade_data(trade_data_1, True)
    calculate_quant_stats(trade_data_1, True)
    # we check the strategy of buying whenever elon tweets about doge
    temp2 = add_allocation(stock_data, tweet_data, False)
    t_test_tweet_volatility(temp2, tweet_data, False)
    trade_data_2 = calculate_profit(temp2)
    plot_trade_data(trade_data_2, False)
    calculate_quant_stats(trade_data_2, False)


def t_test_tweet_volatility(stock_data, tweet_data, only_positive):
    """
    This method compares the volaitility when Elon tweets
    with the volaitlity when he doesn't 
    it plots both dustributions and does a t test
    """
    # we set up a figure with 2 plots
    fig, ax = plt.subplots(2, figsize=(10, 12))
    # we add a daily return column which is the pct change of a stock
    stock_data['Daily Return'] = stock_data['Open'].pct_change()
    # we filter our data into when he tweets vs when he does not
    with_tweet = stock_data[stock_data['signal'] == 1]
    without_tweet = stock_data[stock_data['signal'] == 0]
    # we plot the 2 histograms and label them
    ax[0].hist(with_tweet['Daily Return'].dropna(),
               bins=20,
               label="With Tweets")
    ax[1].hist(without_tweet['Daily Return'].dropna(),
               bins=100,
               label="Without Tweets")
    ax[0].set_xlabel("Volatility")
    ax[1].set_xlabel("Volatility")
    ax[0].set_ylabel("Frequency")
    ax[1].set_ylabel("Frequency")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlim([-0.5, 0.5])
    # we get the statistics
    t_stat, p = stats.ttest_ind(without_tweet['Daily Return'].dropna(),
                                with_tweet['Daily Return'].dropna(),
                                equal_var=False)
    # we finish plotting and conduct our statsitcal tests,
    # with different output depending on the only_positive vondition
    if only_positive:
        ax[0].set_title(
            'Volatility of Dogecoin with Positive Elon Musk Tweets')
        ax[1].set_title(
            'Volatility of Dogecoin without Positive Elon Musk Tweets')
        plt.tight_layout()
        plt.savefig(
            'Volatility of Dogecoin with and without Positive Elon Musk Tweets.png'
        )
        print('Mean volatility When Elon Musk Tweets Positively: ' +
              str(with_tweet['Daily Return'].dropna().mean()))
        print("Mean volatility When Elon Musk Doesn't Tweet Positively: " +
              str(without_tweet['Daily Return'].dropna().mean()))
        print('T-Statistic: ' + str(t_stat))
        print(
            "Because our p-value (" + str(p) +
            ") is greater than alpha = 0.05, we fail to reject the null hypothesis. There isn't convincing statistical evidence that the mean volatility of dogecoin when Elon Musk tweets positively is different from the mean volatility of dogecoin when Elon Musk does not tweet positively"
        )
    else:
        ax[0].set_title('Volatility of Dogecoin with Elon Musk Tweets')
        ax[1].set_title('Volatility of Dogecoin without Elon Musk Tweets')
        plt.tight_layout()
        plt.savefig(
            'Volatility of Dogecoin with and without Elon Musk Tweets.png')
        print('Mean volatility When Elon Musk Tweets: ' +
              str(with_tweet['Daily Return'].dropna().mean()))
        print("Mean volatility When Elon Musk Doesn't Tweet: " +
              str(without_tweet['Daily Return'].dropna().mean()))
        print('T-Statistic: ' + str(t_stat))
        print(
            "Because our p-value (" + str(p) +
            ") is greater than alpha = 0.05, we fail to reject the null hypothesis. There isn't convincing statistical evidence that the mean volatility of dogecoin when Elon Musk tweets positively is different from the mean volatility of dogecoin when Elon Musk does not tweet "
        )


def calculate_quant_stats(trade_data, only_positive):
    """
    This method takes the trade data and whether or not
    we are using the only positive strategy
    """
    # we coonvert the trade data back to a time series dataframe
    trade_data = trade_data.set_index('Date')
    # we calculate the total strateggy return
    # undoing the log we did when accumulating
    trade_data['strategy_return'] = 100 * np.exp(
        trade_data['strategy_cumulative_log_return'] - 1)
    # we print out what strategy we are using
    if only_positive:
        print("Positive Results:")
    else:
        print("Full Results:")
    # we print out our trade data


def plot_trade_data(trade_data, only_positive):
    """
    This method creates a plot which represents
    the difference in profitability between holding 
    doge and trading off of the tweets by using a 1% trailing
    stop loss 
    """
    # we convert the trade data to a time series dataframe
    trade_data = trade_data.set_index('Date')
    # we undo the log and convert it into a percent for
    # the strategy and hold percentages
    trade_data['strategy_cumulative_pc_returns'] = 100 * (
        np.exp(trade_data['strategy_cumulative_log_return']) - 1)
    trade_data['hold_cumulative_pc_returns'] = 100 * (
        np.exp(trade_data['hold_cumulative_log_return']) - 1)
    fig, ax = plt.subplots(figsize=(15, 7))
    # we plot the holding and strategy plots
    ax.plot(trade_data['hold_cumulative_pc_returns'], label="holding method")
    ax.set_ylabel('Cummulative % return')
    ax.set_xlabel('Time')
    ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1))
    if only_positive:
        ax.plot(trade_data['strategy_cumulative_pc_returns'],
                label='positive elon tweet method')
        ax.set_title('Holding vs Positive Strategy Returns')
        plt.savefig('Holding vs Positive Strategy Returns.png')
    else:
        ax.plot(trade_data['strategy_cumulative_pc_returns'],
                label='full elon tweet method')
        ax.set_title('Holding vs Full Strategy Returns')
        plt.savefig('Holding vs Full Strategy Returns.png')


def calculate_profit(stock_data):
    """
    Calculates the profit for holding and the strategy
    using the input signals, exit signals, portfolio allocation,
    and price data 
    """
    # for each row in the stock_data dataframe
    for i, row in stock_data.iterrows():
        # if this is the first row
        if i == 0:
            # we want the current returns to be NaN
            # and the cummulative rows to start at 0
            hold_log_return = np.nan
            hold_cumulative_log_return = 0
            strategy_log_return = np.nan
            strategy_cumulative_log_return = 0
            # we preserve the row allocation data
            allocation = row['allocation']
        # if not the first row
        else:
            # we get the previous value
            prev_open = stock_data.loc[i - 1, 'Open']
            # we make the return for this day to be the log of the current over the
            # previous
            hold_log_return = np.log(row['Open'] / prev_open)
            # we get the previosuly held cummulative sum
            prev_hold_cumulative_return = stock_data.loc[
                i - 1, 'hold_cumulative_log_return']
            # we add the current one to the current total
            hold_cumulative_log_return = prev_hold_cumulative_return + hold_log_return
            # we get the previous allocation
            prev_allocation = stock_data.loc[i - 1, 'allocation']
            # our current return is the allocation (0 for
            # not owned, 1 for owned) times the current hold profit
            strategy_log_return = prev_allocation * hold_log_return
            # we add the current return to the total
            prev_cumulative_return = stock_data.loc[
                i - 1, 'strategy_cumulative_log_return']
            strategy_cumulative_log_return = prev_cumulative_return + strategy_log_return
            # we preserve the row allocation data
            allocation = row['allocation']
        # modify the existing rows in the dataframe with the new data
        stock_data.loc[i, 'hold_log_return'] = hold_log_return
        stock_data.loc[
            i, 'hold_cumulative_log_return'] = hold_cumulative_log_return
        stock_data.loc[i, 'strategy_log_return'] = strategy_log_return
        stock_data.loc[
            i,
            'strategy_cumulative_log_return'] = strategy_cumulative_log_return
        stock_data.loc[i, 'allocation'] = allocation
    # create a new dataframe with the relevant data from the stock data
    trade_data = stock_data[[
        'Date', 'Open', 'hold_log_return', 'hold_cumulative_log_return',
        'strategy_log_return', 'strategy_cumulative_log_return'
    ]]
    # return that new dataframe
    return trade_data


def example_trade(stock_data, tweet_data):
    """
    This method plots a graph representing an entry and 
    exit of a trade by drawing the price line and trailing 
    stop loss line 
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    # we resammple the tweet data to daily data
    tweet_data = make_daily(tweet_data)
    temp = stock_data.loc[tweet_data.iloc[0]['Timestamp']:tweet_data.
                          iloc[0]['Timestamp'] + pd.Timedelta(days=5)]
    # we reset back to a normal dataframe so we dont plot the dates at the bottom
    temp = temp.reset_index()
    # We calculate the stop price from the cummulative max (the maxmium price)
    # we have seen so far times 0.99 (1 - the callback percentage/100)
    # our callback percentage is 1
    # essentially, we hold until we go 1% below the maximum we have seen so far
    temp['stop price'] = temp['Open'].cummax() * 0.95
    # Our exit signal is when we go below the stop price, with 0 meaning below
    # and 1 being above
    temp['exit signals'] = (temp['Open'] < temp['stop price']).astype(int)
    # we get our first exit point (idx max returns the first one we have)
    exit_point = temp.loc[temp['exit signals'].idxmax()]
    # we plot the 2 graphs
    ax.plot(temp['Open'], label="current price")
    ax.plot(temp['stop price'], label='stop price')
    # we plot the exit point (current date differency from the tweet date in days)
    ax.scatter((exit_point['Date'] - temp.loc[0]['Date']).days,
               exit_point['Open'],
               c='r',
               marker='o')
    ax.set_ylabel('DOGE Price')
    ax.set_xlabel('Time since tweet (days)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1))
    ax.set_title('Example Trade')
    plt.savefig('Example Trade.png')


def add_allocation(stock_data, tweet_data, only_positive):
    """    
    This method adds an allocation of doge to your portfolio 
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    tweet_data = make_daily(tweet_data)
    # Will only add allocation for doge if you only want to consider
    # Positive sentiment tweets
    if only_positive:
        tweet_data = tweet_data[tweet_data['sentiment'] > 0.0]
    stock_data = stock_data.reset_index()
    # Converts boolean series into 1 or 0 using astype(int) to determine entry dates
    stock_data['signal'] = stock_data['Date'].isin(
        tweet_data['Timestamp']).astype(int)
    # Calculates stop loss
    stock_data['stop price'] = stock_data['Open'].cummax() * 0.95
    stock_data['exit signals'] = (stock_data['Open'] <
                                  stock_data['stop price']).astype(int)
    stock_data['allocation'] = 0
    # we get all the entry indexes and exit indexes
    entry_indexes = stock_data[stock_data['signal'] == 1].index
    exit_indexes = stock_data[stock_data['exit signals'] == 1].index
    # create a new series representing whether or not we are in a trade (0 being
    # no trade and 1 being trade)
    in_trade = pd.Series(0, index=stock_data.index)
    result = []
    for entry_signal in entry_indexes:
        # we loop through every entry signal
        for exit_signal in exit_indexes:
            # if our entry signal is greater than and exit signal, we skip
            # the next lines
            if exit_signal <= entry_signal:
                continue
            # we append the entry and exit indexes of a trade as tuple
            # to our list
            result.append((entry_signal, exit_signal))
            # we skip to the next exit signals to avoid overlap
            break
    # we set the allocation to be 1 in between the given entry and exit signals
    for entry, exit in result:
        in_trade.loc[entry:exit] = 1
    stock_data['allocation'] = in_trade
    return stock_data


def plot_volatility(stock_data, tweet_data):
    """
    This method plots the percentage change in dogecoin for a given 
    tweet over a 5 day time horizon (post tweet) using a line plot 
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    # we convert tweet data to a regular dataframe and remove the extra time data
    tweet_data = make_daily(tweet_data)
    for i in range(len(tweet_data)):
        # For every tweet, we get the stock price 5 days before
        # and 5 days after
        temp = stock_data.loc[tweet_data.iloc[i]['Timestamp'] - pd.Timedelta(
            days=5):tweet_data.iloc[i]['Timestamp'] + pd.Timedelta(days=5)]
        temp = temp.reset_index()
        # we calculate the percentage change before and after he tweets
        # with 0% change being the time he tweeted at
        temp['percentage change'] = temp['Open'] / temp.loc[5][
            'Open'] * 100 - 100
        # we plot the days before and after the tweet on the x axis
        # and the percentage change on the y
        ax.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                temp['percentage change'])
    ax.set_xlabel("Time since tweet (Days)")
    ax.set_ylabel("Percentage Change of Doge Value")
    ax.set_title('Volatility Before and After Elon Tweets')
    plt.savefig('Volatility Before and After Elon Tweets.png')


def plot_r_squared(stock_data, tweet_data):
    """ 
    This method plots a scatter plot from which you can identify 
    the relationship between the sentiment of tweets by elon a given day and the price 
    """
    fig, ax = plt.subplots(1, figsize=(15, 7))
    # Returns merged data which can be plotted
    merged_df = clean_data_plot_1(stock_data, tweet_data)
    # Plot a scatter plot of sentiment on a day with change in price on that day
    ax.scatter(x=merged_df['sentiment'], y=merged_df['Daily Change'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Daily Change")
    # Conducts linear regression on the data using the scipy library
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    m, b, r, p, std_err = stats.linregress(merged_df['sentiment'],
                                           merged_df['Daily Change'])
    # Calculate r^2 stat
    print(
        str(r * r * 100) +
        "% of the daily variablity in dogecoin price is explained by variability in Elon Musk's tweet sentiment toward dogecoin"
    )
    line = m * merged_df['sentiment'] + b
    # Plots line of best fit
    ax.plot(merged_df['sentiment'], line, c='red')
    ax.set_title('Sentiment vs Daily Change')
    plt.savefig('Sentiment vs Daily Change.png')


def clean_data_plot_1(stock_data, tweet_data):
    """
    This method resamples the tweet data to a daily sentiment 
    score and filters negligible datapoints  
    """
    stock_data['Daily Change'] = stock_data['Close'] - stock_data['Open']
    tweet_data = make_daily(tweet_data)
    # Remove no sentiment tweets because those shouldn't move prices of doge
    tweet_data = tweet_data[tweet_data['sentiment'] != 0.0]
    stock_data = stock_data.reset_index()
    # Merge data based on date of tweet with price of doge on that day
    merged = tweet_data.merge(stock_data, left_on='Timestamp', right_on='Date')
    merged = merged[['sentiment', 'Daily Change']]
    return merged


def get_data(name1, name2):
    """ 
    This method retrieves the data and returns 
    the filtered tweet dataframe
    """
    stock_data, tweet_data = load_data(name1, name2)
    tweet_data = filter_data(tweet_data)
    return stock_data, tweet_data


def filter_data(tweet_data):
    """
    This method filters each tweet and applies a sentiment
    score to the tweets by using the nltk sentiment analyzer 
    """
    tweet_data = tweet_data[['text']]
    # Filter data down to 2018-2022 for Elon because that's all
    # we have for dogecoin data
    tweet_data = tweet_data.loc['2018':'2022']
    # applies clean_tweet method to all tweets which removes all noise
    # leaving only the actual content of the tweet
    tweet_data['text'] = tweet_data['text'].apply(clean_tweet)
    # Filter to tweets where doge is actually mentioned
    mask = tweet_data['text'].str.contains("doge")
    tweet_data = tweet_data[mask]
    # Finally get a sentiment score for each tweet now that the text
    # column only holds the tweet
    tweet_data['sentiment'] = tweet_data['text'].apply(get_sentiment)
    return tweet_data


def get_sentiment(tweet):
    """
    This method gets a sentiment score between -1 
    (Very negative) and 1 (very positive) for each tweet where "doge" 
    is mentioned 
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(tweet)
    return scores['compound']


def clean_tweet(tweet):
    """
    This method filters tweet data using a multidude of regex expressions 
    and standardizing tweets to lower case 
    """
    # removes the extra 3 tokens at the end (Likes, comments, retweets info)
    tokens = tweet.split()
    tokens = tokens[0:len(tokens) - 3]
    tweet = ' '.join(tokens)
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    # Found the url regex in the link above
    tweet = re.sub(r'https\S+', ' ', tweet)
    # removes tabs and double spaces, makes it lowercase
    tweet = re.sub(r'[\n\t]', ' ', tweet)
    # removes double spaces
    tweet = re.sub('\s\s+', ' ', tweet)
    # return raw text of tweet in lowercase for standardization purposes
    return tweet.lower()


def make_daily(tweet_data):
    """
    This method finds the mean sentiment 
    score of Elon's tweets that mention "doge"
    on a given day 
    """
    tweet_data = tweet_data[['sentiment']]
    tweet_data = tweet_data.resample('D').mean().dropna()
    # we convert it back to a normal dataframe
    tweet_data = tweet_data.reset_index()
    tweet_data['Timestamp'] = tweet_data['Timestamp'].dt.tz_localize(None)
    return tweet_data


def load_data(name1, name2):
    """
    This method converts our CSV files into time series dataframes 
    """
    # Stock data time series by date
    stock_data = pd.read_csv(name1, index_col="Date", parse_dates=True)
    stock_data = stock_data.sort_index()
    # Tweet data indexed by minute
    tweet_data = pd.read_csv(name2, index_col="Timestamp", parse_dates=True)
    tweet_data = tweet_data.sort_index()
    return stock_data, tweet_data


if __name__ == '__main__':
    main()
