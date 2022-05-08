import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/37354105/find-the-end-of-the-month-of-a-pandas-dataframe-series
from pandas.tseries.offsets import MonthEnd, YearEnd

def snake_case_columns(df):
    # snake case column names
    # https://stackoverflow.com/questions/19726029/how-can-i-make-pandas-dataframe-column-headers-all-lowercase
    df.columns = map(lambda x : x.lower().replace(" ", "_"), df.columns)


def beef_price_lineplot(title, beef_data):
    # https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot
    fig, ax = plt.subplots(figsize=(18, 14))
    # https://stackoverflow.com/questions/60936733/how-to-set-x-axis-in-every-10-years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))

    sns.lineplot(ax = ax, x='month', y='value', hue='type_of_cattle', 
                 data=beef_data).set_title(title)

    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
    ax.yaxis.set_major_locator(ticker.MaxNLocator())

    plt.xlabel('Date')
    plt.ylabel('Price per 100 Kg (EUR)')

    plt.show()

def cso_date_to_datetime(df, date_col_name):
    df[date_col_name] = pd.to_datetime(df[date_col_name], format='%YM%m') + MonthEnd(1)

def transform_indexmundi_yearly_data(df, col_name_value):
    df["Market Year"] = pd.to_datetime(df["Market Year"], format='%Y') + YearEnd(1)
    df = df.rename(columns={"Market Year" : "month", " Value" : col_name_value})
    df = df.drop([" Unit Description"], axis=1)
    df = df.set_index("month")
    df = df.resample('M').last().bfill() / 12
    df = df[df.index.year > 1989]
    df = df[df.index.year < 2022]
    return df

def prepare_forex_data(file_path):
    df_cur = pd.read_csv(file_path)
    snake_case_columns(df_cur)
    
    # https://dataindependent.com/pandas/pandas-to-datetime-string-to-date-pd-to_datetime/
    df_cur["date"] = pd.to_datetime(df_cur["date"], format='%b %y') + MonthEnd(1)
    # only need final day closing price
    df_cur = df_cur.drop(["open", "high", "low", "change_%"], axis=1) 
    df_cur = df_cur.set_index("date")
    df_cur = df_cur.sort_index()
    return df_cur

def transform_fred_stlouisfed_quarterlydata(df, column_index_name):
    df["DATE"] = pd.to_datetime(df["DATE"], format='%Y-%m-%d') + MonthEnd(1)
    
    df = df.rename(columns= { df.columns[1] : column_index_name })
    snake_case_columns(df)
    
    df = df.set_index("date")
    df = df.resample('M').last().bfill()

    df_missing = pd.date_range(start=df.tail(1).index[0], end='31-DEC-2021', freq='M').to_frame(index=False, name='date')[1:] # cut first date off since it will be the last date of the real dataframe
    df_missing[column_index_name] = np.nan
    df_missing = df_missing.set_index("date")


    df = pd.concat([df, df_missing])
    df[column_index_name] = df[column_index_name].interpolate(method='linear', limit_direction='both')

    # df = df.resample('M').last().bfill()
    df = df[df.index.year > 1989]
    
    return df