import matplotlib.pyplot as plt
import pandas as pd

event_file = ''
event_df = pd.read_csv(
	'/workspace/data/prediction_catalog_final_loc.dat',
	sep=r'\s+',
	header=None,
	skiprows=1,
	names=['time', 'x', 'y', 'z', 'MomMag', 'Location', 'label2'],
	na_values='N/A',
	engine='python',
)
event_df['time'] = pd.to_datetime(event_df['time'])

image_dir = 'workspace/data/images/Yu2024_catalog'

event_df['date'] = event_df['time'].dt.date

print(event_df.head())
print('number of events:', len(event_df))


# 日ごとにカウント
daily_counts = event_df.groupby('date').size()

# プロット
daily_counts.plot(kind='bar', figsize=(10, 5))
plt.title('Event count per day')
plt.xlabel('Date')
plt.ylabel('Number of events')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(f'{image_dir}/event_count_per_day.png')

# NaN を除外したマグニチュードのみ対象
magnitudes = event_df['MomMag'].dropna()

plt.figure(figsize=(8, 5))
plt.hist(magnitudes, bins=20, edgecolor='black')  # binsは適宜調整
plt.title('Histogram of Event Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
