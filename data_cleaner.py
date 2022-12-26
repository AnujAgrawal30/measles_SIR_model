import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

df['Date case visited -1'] = pd.to_datetime(df['Date case visited -1'], dayfirst=True)
df.set_index('Date case visited -1', inplace=True)
daily_count = df.groupby(pd.Grouper(freq='D')).size()
daily_count = daily_count.resample('D').last().fillna(0)


# Group the data by the date column and compute the size of each group
# daily_count = df.groupby(['Date case visited -1']).size()

# Generate a sequence of dates to use as the new index
# dates = pd.date_range(start=daily_count.index.min(), end=daily_count.index.max())

# Reindex the daily count with the new dates
# daily_count = daily_count.reindex(index=dates, fill_value=0)

# Print the resulting daily count
print(list(daily_count))
