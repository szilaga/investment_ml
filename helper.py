from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
from datetime import timedelta
from get_all_tickers import get_tickers as gt
import ipywidgets as widgets

class Helper:

    def getParameters(self):

        forecast = widgets.Text(
            value='7;14;30',
            placeholder='Type something',
            description='Forecasts:',
            disabled=False
        )

        tickers = widgets.Text(
            value='GOOG;NOK',
            placeholder='Type something',
            description='Tickers:',
            disabled=False
        )

        start_date = widgets.DatePicker(
            description='Start Date:',
            disabled=False
        )

        end_date = widgets.DatePicker(
            description='End Date:',
            disabled=False,
            value=pd.Timestamp.today(tz=None))

        r_std = widgets.IntSlider(value=30, min=0, description='Rolling std:')
        r_sma1 = widgets.IntSlider(value=30, min=0, description='SMA 1:')
        r_sma2 = widgets.IntSlider(value=50, min=0, description='SMA 2:')
        r_rsi = widgets.IntSlider(value=14, min=0, description='RSI:')

        return r_std,r_sma1,r_sma2, r_rsi,start_date, end_date, forecast, tickers


    #### Helper functions
    def get_Tickers_Yahoo(self):
        '''
        This functions querys all available tickers of Yahoo
        :return: list of ticker symbols
        '''
        return gt.get_tickers()


    def get_Data_Yahoo(self,stock_code, start_date, end_date):
        '''
        Get data from yfinance
        https://pypi.org/project/yfinance/
        :param stock_code:
        :param start_date:
        :param end_date:
        :return: Dataframe
        '''
        # Data fetching
        return pdr.get_data_yahoo(stock_code, start_date, end_date)

    def cleanDataFrame(self,df):
        '''
        Clean dataframe by following features
        FrontFill if by missing dates
        :return: Dataframe
        '''

        # create dataframe with entire timespan
        df_time = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1]))
        # join with extracted dataframe
        df_time = df_time.join(df, how='left')
        # perform frontfill
        df_time.ffill(axis=0, inplace=True)

        return df_time

    def get_nullValues_ext(self, df, axis=0):
        '''
        The function counts the number of nan values per column\n",
        and the total number of nan\n",
        Note: axis = 0 -> row wise; axis = 1 column wise\n",
        '''
        return df.isnull().sum(axis=axis).to_frame()

    def normalize(self, df):
        '''
        normalize all values of a dataframe between 0 and 1
        :param df:
        :return: Dataframe, min, max
        '''
        # normalize values
        return (df - df.min()) / (df.max() - df.min()),df.max(),df.min()

    def denormalize(self,df,max,min):
        '''
        recreate dataframe before normalization
        :param df:
        :param max:
        :param min:
        :return:Dataframe
        '''
        return df * (max - min) + min

    def normalize_price(self, data):
        '''
        returns normalized price
        return: dataframe
        '''

        n_price = data / data.iloc[0]

        return pd.DataFrame(n_price)

    def slice_forward(self, df, sample):
        '''
        forward slice of dataframe by sample
        return: dataframe
        '''
        days = timedelta(sample)
        start_date = df.index[0]
        end_date = df.index[0] + days

        return slice_df(df, start=start_date, end=end_date)

    def slice_backward(self, df, sample):
        '''
        backward slice of dataframe by sample
        return: dataframe
        '''
        days = timedelta(sample)
        start_date = df.index[-1] - days

        return slice_df(df, start=start_date, end=df.index[-1])

    def slice_df(self, df, start='2021-01-01', end='2021-06-04'):
        '''
        slices the dataframe along the y-axis
        respective the rows (dates)
        return: dataframe
        '''
        return df.loc[start:end]

    def nth_root(self, num, root):
        '''
        calculte the nth root of a numer
        return: float
        '''
        return num ** (1 / root)

    def get_SplitData(self, df, train_pct):
        '''
        split the dataset into train and test data
        :param train_pct:
        :return: Dataframes: x_train, y_train, x_test, y_test, train, tes
        '''
        # train_pct is percentage for training dataset

        train_pt = int(df.shape[0] * train_pct)

        # extract train dataset
        train = df.iloc[:train_pt, :]
        # extract test dataset
        test = df.iloc[train_pt:, :]

        x_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]

        x_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        return x_train, y_train, x_test, y_test, train, test

    def get_SplitData_(self, df, n_forecast, train_pct, ticker):
        '''
        Different approach: Split data into train and test dataset
        :param n_forecast:
        :param train_pct:
        :param ticker:
        :return: Dataframes
        '''
        # train_pct is percentage for training dataset

        train_pt = int(df.shape[0] * train_pct)

        # extract train dataset
        train = df.iloc[:train_pt, :]
        # extract test dataset
        test = df.iloc[train_pt:, :]

        test = slice_forward(test, n_forecast)

        x_train = train.iloc[:, 1:]
        y_train = train[ticker]

        x_test = test.iloc[:, 1:]
        y_test = test[ticker]

        return x_train, y_train, x_test, y_test, train, test

    def set_shift(self, data, forecast, column):
        '''
        This function shifts the stockprice according to the length,
        the stock shall be predicted
        :param data:
        :param forecast:
        :param column:
        :return: Dataframe
        '''
        # extract price forecast
        shift = np.array(data[column].shift(periods=-forecast, axis=0))[:-forecast]
        # set index to column
        data['date'] = data.index
        # shit column date
        data['date'] = data['date'].shift(periods=-forecast, axis=0)
        # cut last columns
        data.drop(data.tail(forecast).index, inplace=True)
        # set column to index
        #data = data.set_index('date')
        # drop date column
        data.drop('date', axis=1, inplace=True)
        # add forecast values
        data['{}_shift'.format(column)] = shift

        return data

