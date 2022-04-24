import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd

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
    # https://stackoverflow.com/questions/37354105/find-the-end-of-the-month-of-a-pandas-dataframe-series
    from pandas.tseries.offsets import MonthEnd
    df[date_col_name] = pd.to_datetime(df[date_col_name], format='%YM%m') + MonthEnd(1)