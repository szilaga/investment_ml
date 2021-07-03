import pandas as pd
import numpy as np
from helper import Helper

class Statistics():

    #### Statistical functions
    def get_globalStats(self, data, column):
        '''
        calculate global statistical numbers
        return: mean (float), median (float), standard deviation (float)
        '''
        mean = data[column].mean()
        median = data[column].median()
        std = data[column].std()

        return round(mean, 4), round(median, 4), round(std, 4)

    def get_sma(self, data, n):
        '''
        calculates the simple moving average of the stock price
        return: dataframe
        '''
        sma = data.rolling(window=n).mean()
        return pd.DataFrame(sma)

    def get_rstd(self, data, n):
        '''
        calculates the rolling standard deviation
        return: dataframe
        '''
        rstd = data.rolling(window=n).std()
        return pd.DataFrame(rstd)

    def get_BollingerBand(self, sma, rstd):
        '''
        calculates the bollinger bandwith
        return: series upper & series lower
        '''
        # upper bound
        upper_band = sma + (rstd * 2)
        # lower bound
        lower_band = sma - (rstd * 2)

        return upper_band, lower_band

    def get_DailyReturn(self, data):
        '''
        calculates the daily return of the stockprice in percent
        return: dataframe
        '''
        # p_today / p_yesterday - 1
        d_ret = round((data[1:] / data[:-1].values) - 1, 4)

        return pd.DataFrame(d_ret)  # .bfill(axis = 0,inplace=True)

    def get_CumulativeReturn(self, data):
        '''
        calculates the price development since the beginning
        of the records
        '''
        # p_today / p_begin -1

        d_retcom = round((data / data.iloc[0]) - 1, 4)

        return pd.DataFrame(d_retcom)

    def get_Tickerprice(self, price, low, high):
        '''
        calculates the ticker price
        return: Dataframe
        '''
        ticker = (price + low + high) / 3
        return pd.DataFrame(ticker)

    def get_Momentum(self, df, column, n):
        '''
        calcualte the momentum of the stock
        return: Dataframe
        '''
        df['ret'] = get_daily_return(df[column])
        return df['ret'].pct_change(n)

    def get_RSI(self, df, price, n):
        '''
        calculate RSI value
        return: Dataframe
        '''
        df = df.copy()

        df['delta'] = df[price] - df[price].shift(1)
        df['gain'] = np.where(df['delta'] >= 0, df['delta'], 0)
        df['loss'] = np.where(df['delta'] < 0, abs(df['delta']), 0)
        avg_gain = []
        avg_loss = []
        gain = df['gain'].tolist()  # transforming the column into a numpy list
        loss = df['loss'].tolist()

        for i in range(len(df)):  # loop
            if i < n:
                avg_gain.append(np.NaN)
                avg_loss.append(np.NaN)
            elif i == n:
                avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
                avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
            elif i > n:
                avg_gain.append(((n - 1) * avg_gain[i - 1] + gain[i]) / n)
                avg_loss.append(((n - 1) * avg_loss[i - 1] + loss[i]) / n)

        df['avg_gain'] = np.array(avg_gain)
        df['avg_loss'] = np.array(avg_loss)
        df['RS'] = df['avg_gain'] / df['avg_loss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))
        return df['RSI']

    def get_Sharprisk(self, data, k):
        '''
        calculate the sharp risk rate for a stockcode
        return: float as percent

        k:
        daily = 252
        weekly = 52
        monthly = 12

        Note:
        The greater a portfolio's Sharpe ratio, the better its risk-adjusted-performance.
        If the analysis results in a negative Sharpe ratio, it either means the risk-free rate
        is greater than the portfolio’s return, or the portfolio's return is expected to be negative.
        In either case, a negative Sharpe ratio does not convey any useful meaning.

        '''
        h = Helper()

        # get daily riskfree
        d_rf = round(h.nth_root(1.0 + 0.1, 252) - 1, 4)

        # get daily return
        d_rt = data.to_frame()

        # daily return minus riskfree
        d_rt['rt-rf'] = d_rt - d_rf

        # equation sharp risk rate
        sr = np.sqrt(k) * (d_rt['rt-rf'].mean() / d_rt[d_rt.columns[0]].std())

        return round(sr, 3)