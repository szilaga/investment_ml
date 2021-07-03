# investment_ml
Investment_ml is the idea to evaluate different ML approaches to predict stockprices based on different statistical values and historical price data
The data set is directly query over the yfinance API.


## Table of contents

- [Quickstart](#quick-start)
- [Idea](#idea)
- [What's included](#whats-included)
- [Description](#description)
- [Copyright and license](#copyright-and-license)



# Quickstart
Download the entire Git Repo including the datasets. Execute the notebook <b>"stock_predictor.ipynb"</b> in your jupyter.<br/>
Before execution: If you do not already have installed the listed libraies below, please install them in a previous step.

- Install <a href="https://pypi.org/project/pandas/">Pandas:</a> `pip install pandas`
- Install <a href="https://pypi.org/project/numpy/">Numpy:</a> `pip install numpy`
- Install <a href="https://pypi.org/project/matplotlib"/>Plotlib:</a> `pip install matplotlib` 
- Install <a href="https://pypi.org/project/seaborn">Seaborn:</a> `pip install seaborn`
- Install <a href="https://pypi.org/project/yfinance/">yfinance:</a> `pip install yfinance`
- Install <a href="https://pypi.org/project/plotly/">Plotly:</a> `pip install plotly`
- Install <a href="https://pypi.org/project/scipy/">Scipy:</a> `pip install scipy`
- Install <a href="https://pypi.org/project/scikit-learn/">Scikit:</a> `pip install scikit-learn`
- Install <a href="https://pypi.org/project/pandas-datareader/">Pandas-datareader:</a> `pip install pandas-datareader`


## Idea
The given notebook should for once, test the capabilites of modern ml-algorithm and also gives some insides, if a given stock 
is interesting to trade with. Because of that, the project also includes some statistical key performance indicators.
In future, the notebook shall provide more statistical features to analyse a given stock.

## What's included
```text
investment_ml/
├── stock_price_predictor.ipynb
├── helper.py
├── mllib.py
├── indicators.py
├── plots.py
├── README.MD
```

## Description

#### Set Parameters:
Based on jupyter notebook widgets, the user need the set some parameters before he can continue if the stock analysis.
1. Set some ticker values (https://de.finance.yahoo.com/)
2. Set start and end date to grep data
3. Set days for forecast
4. Set statistical features like simple moving average etc.

#### Prepare Data:
The next section within the notebook perfroms some data preparations like adding weekend days and fills empty tuples by front fill method.
Afterwards statistical values will which will be used as training features.

#### Analyse Datasets:
In this section provides a bunch of statistical values like "sharp risk", "mean","median" and some plots like stock trends or histograms of the daily returns.

#### Perform forecasting:
The last section prepares the actual training and test datasets and performs a stock prediction based on the given ml models.

## Contributing
If you like this project, please feel free to code. 

## Copyright and license
The code itself is free licensed. All libaries, used in this project are still under the given licence model.
For more information please check the licence @pypi.org for every library listed above.


