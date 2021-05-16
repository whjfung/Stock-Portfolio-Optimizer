# Description: This program is inspired by Youtube channel Computer Science at https://www.youtube.com/watch?v=bvDkel5whUY
#              It extracts live > 10Y historical data from Yahoo Finance and produces an optimized stock portfolio from S&P 500
#              using the efficient frontier.

# DISCLAIMER
# This program is for informational purposes only.  Use of and access to this program and the information, materials, services, and 
# other content available on or through this program (“Content”) are subject to these terms of use and all applicable laws.

# NO INVESTMENT ADVICE
# The Content is for informational purposes only, you should not construe any such information or other material as legal, tax, 
# investment, financial, or other advice. Nothing contained on our Site constitutes a solicitation, recommendation, endorsement, 
# or offer by whjfung or any third party service provider to buy or sell any securities or other financial instruments in this or in 
# in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction.
# All Content on this site is information of a general nature and does not address the circumstances of any particular individual 
# or entity. Nothing in this program constitutes professional and/or financial advice, nor does any information on this program 
# constitute a comprehensive or complete statement of the matters discussed or the law relating thereto. whjfung is not a fiduciary 
# by virtue of any person’s use of or access to this program or Content. You alone assume the sole responsibility of evaluating the 
# merits and risks associated with the use of any information or other Content on this program before making any decisions based on 
# such information or other Content. In exchange for using this program, you agree not to hold whjfung, its affiliates or any third 
# party service provider liable for any possible claim for damages arising from any decision you make based on information or other 
# Content made available to you through this program.

# INVESTMENT RISKS
# There are risks associated with investing in securities. Investing in stocks, bonds, exchange traded funds, mutual funds, and 
# money market funds involve risk of loss.  Loss of principal is possible. Some high risk investments may use leverage, which 
# will accentuate gains & losses. Foreign investing involves special risks, including a greater volatility and political, economic 
# and currency risks and differences in accounting methods.  A security’s or a firm’s past investment performance is not a guarantee 
# or predictor of future investment performance.

#***********************************************************************************************************************************

#pip install yfinance

import pandas_datareader as web
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io

portfolio_val = int(input("\nHow much do you want to invest in? "))

print("\nDownloading Data.....\n")

# Get Data from NASDAQ
start = datetime.datetime(2010,1,1)
end = datetime.datetime.now()

url="https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_csv/data/e613177765e570e43c2a1e8330bf73bf/constituents_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))

Stocks = companies['Symbol'].tolist()

Stocks = [stock.replace('.', '-') for stock in Stocks]

df = yf.download(Stocks,start,end)["Adj Close"]

df = df.dropna(axis=1, how='all')

assets = df.columns

#pip install PyPortfolioOpt

# Optimize the Portfolio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Calculate the expected annualised returns and the annualized sample covariance matrix of the daily asset returns
exp_r = expected_returns.mean_historical_return(df)
sam_c = risk_models.sample_cov(df)

# Optimize for miximal Sharpe ratio
ef = EfficientFrontier(exp_r, sam_c)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

#pip install pulp

# Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, portfolio_val)
allocation, leftover = da.lp_portfolio()
#print('Discrete allocation: ', allocation)
#print('Remaining Cash: $', leftover)

# Function that gets company name from its ticker
def get_name(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

# Get company names
company_list = []
for symbol in allocation:
  company_list.append(get_name(symbol))

# Get discrete allocation Values
discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))

# Create dataframe for portfolio
portfolio_df = pd.DataFrame(columns = ['Company name', 'Ticker', 'No. of shares to hold'])

portfolio_df['Company name'] = company_list
portfolio_df['Ticker'] = allocation
portfolio_df['No. of shares to hold'] = discrete_allocation_list

# Show the portfolio
print("\n\n ************ Recommended Portfolio for USD$",portfolio_val,"************\n\n")
print(portfolio_df, "\n")
print('Remaining Cash: $', leftover, "\n")
