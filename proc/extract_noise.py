# %%
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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
tdms_files = np.sort(list(Path(data_dir).glob('*.tdms')))

event_times = pd.to_datetime(event_df['time'])

# 時刻を抽出する正規表現パターン
pattern = re.compile(r'UTC_(\d{8})_(\d{6})')

# イベントを含まないTDMSファイルのリスト
no_event_files = []
event_files = []
for path in tdms_files:
	m = pattern.search(path.name)
	if not m:
		continue  # パースできなかった場合はスキップ

	# ファイルの開始時刻
	dt_str = m.group(1) + m.group(2)  # '20220417' + '034611'
	t_start = datetime.strptime(dt_str, '%Y%m%d%H%M%S').replace(
		tzinfo=pd.Timestamp.utcnow().tz
	)  # UTC aware

	# 終了時刻（15秒後と仮定）
	t_end = t_start + timedelta(seconds=15)

	mask = (event_times >= t_start) & (event_times < t_end)
	# この時間範囲にイベントが1つでも含まれるか？
	if not mask.any():
		no_event_files.append(path)
	else:
		print(f'Event found in file: {path.name}')
		print(f'detected event time: {event_times[mask].values}')
		event_files.append(path)

resample_dir = Path(data_dir) / 'raw_78B_npy'
for _ in range(len(event_times)):
	file = np.random.choice(event_files)
	file = file.stem + '_1kHz.npy'
	data = np.load(resample_dir / file)
	sys.exit()
