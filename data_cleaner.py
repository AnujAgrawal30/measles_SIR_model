import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

df['Date case visited -1'] = pd.to_datetime(df['Date case visited -1'], dayfirst=True)
df.set_index('Date case visited -1', inplace=True)
daily_count = df.groupby(pd.Grouper(freq='D')).size()
daily_count = daily_count.resample('D').last().fillna(0)

print(list(daily_count))
