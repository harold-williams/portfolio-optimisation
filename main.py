import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import h5py
import scipy.optimize as sco

cached = False

table = None

if not cached:
    quandl.ApiConfig.api_key = '17cy-QeTvNNS-qR6Exw1'
    stocks = ['FB', 'AAPL','AMZN', 'NFLX', 'GOOGL']
    data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': '2017-1-1', 'lte': '2022-12-31' }, paginate=True)


    df = data.set_index('date')
    table = df.pivot(columns='ticker')
    # By specifying col[1] in below list comprehension
    # You can select the stock names under multi-level column
    table.columns = [col[1] for col in table.columns]
        
    table.to_hdf('FAANG.hdf5', key='faang_data', mode='w')
    
    print(table.head())

else:
    table = pd.read_hdf('FAANG.hdf5', 'faang_data')

    print(table)



plt.figure(figsize=(14, 7))
for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')
plt.show()
