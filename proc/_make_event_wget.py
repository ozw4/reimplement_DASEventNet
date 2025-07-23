# %%
import re
from datetime import datetime, timedelta

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

data_dir = '/workspace/data/silixa'

with open(data_dir + '/get_silixa_raw_tdms_april_2022.sh') as f:
	lines = f.readlines()

event_times = pd.to_datetime(event_df['time'])

# 時刻を抽出する正規表現パターン
pattern = re.compile(r'UTC_(\d{8})_(\d{6})')

# イベントを含まないTDMSファイルのリスト
event_line = []
no_event_line = []
for url in lines:
	match = pattern.search(url)
	if not match:
		continue

	dt_str = match.group(1) + match.group(2)  # '20220416' + '163741'
	t_start = datetime.strptime(dt_str, '%Y%m%d%H%M%S').replace(
		tzinfo=pd.Timestamp.utcnow().tzinfo
	)
	t_end = t_start + timedelta(seconds=15)

	# この時間範囲にイベントが1つでも含まれるか
	mask = (event_times >= t_start) & (event_times < t_end)
	if mask.any():
		event_line.append(url)
	else:
		no_event_line.append(url)

with open(data_dir + '/get_silixa_raw_tdms_april_2022_event.txt', 'w') as f:
	for url in event_line:
		f.write(url)

with open(data_dir + '/get_silixa_raw_tdms_april_2022_no_event.txt', 'w') as f:
	for url in no_event_line:
		f.write(url)
print(f'Number of event files: {len(event_line)}')
print(f'Number of no event files: {len(no_event_line)}')
