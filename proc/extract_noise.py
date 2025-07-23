# %%
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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

preprocessed_dir = '/workspace/data/silixa/raw_78B_npy'
preprocessed_files = np.sort(list(Path(preprocessed_dir).glob('*.npy')))


event_times = pd.to_datetime(event_df['time'])

# 時刻を抽出する正規表現パターン
pattern = re.compile(r'UTC_(\d{8})_(\d{6})')

# イベントを含まないTDMSファイルのリスト
no_event_files = []
event_files = []
for path in preprocessed_files:
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
window_size = 2000  # 2秒分のサンプル数（1kHzサンプリング）
extract_noise = []
num_silixa_event = 1309

for i in tqdm(range(num_silixa_event)):
	file = np.random.choice(no_event_files)
	# file = file.stem + '_1kHz.npy'
	data = np.load(resample_dir / file)
	random_idx = np.random.randint(0, data.shape[1] - window_size + 1)
	extract_data = data[:, random_idx : random_idx + window_size].astype(np.float32)
	extract_noise.append(extract_data)
	if i % 100 == 0:
		plt.figure()
		plt.imshow(extract_data, aspect='auto', vmin=-1, vmax=1, cmap='seismic')
		# plt.title(time.strftime('%Y-%m-%d %H:%M:%S'))
		plt.savefig(f'/workspace/image/extract_noise/{i}.png')
		plt.cla()
		plt.clf()
		plt.close()
extract_noise = np.array(extract_noise)
np.save(
	'/workspace/data/extract_noise_2s.npy',
	extract_noise,
)

# compress saved file for colab
np.save(
	'/workspace/data/extract_noise_2s_float16.npy',
	extract_noise.astype(np.float16),
)


np.save(
	'/workspace/data/extract_noise_2s_example.npy',
	extract_noise[:10],
)


# %%
