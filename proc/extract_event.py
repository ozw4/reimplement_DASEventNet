# %%
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_2s_window(
	data: np.ndarray,
	t: datetime,
	time: datetime,
	seed: int | None = None,
	margin: tuple = (100, 1600),
) -> np.ndarray:
	"""DAS 15秒データからイベントを含む2秒間 (2000サンプル) を抽出。
	イベントが [margin[0], margin[1]] の範囲に収まるように切り出す。
	範囲に収まらない場合は margin を無視して通常抽出にフォールバック。

	Parameters
	----------
	data : np.ndarray
	    1D array (長さ15000を想定)
	t : datetime
	    ファイル開始時刻
	time : datetime
	    イベント時刻
	seed : int | None
	    ランダムシード
	margin : tuple[int, int]
	    ウィンドウ内でイベントが来てほしい相対位置（サンプル）

	Returns
	-------
	np.ndarray
	    長さ2000のスニペット

	"""
	if seed is not None:
		np.random.seed(seed)

	sampling_rate = 1000
	total_samples = 15000
	window_size = 2000
	min_margin, max_margin = margin

	offset_sec = (time - t).total_seconds()
	event_idx = int(round(offset_sec * sampling_rate))

	if not (0 <= event_idx < total_samples):
		raise ValueError(f'イベント時刻がファイル範囲外: event_idx={event_idx}')

	# margin を考慮した切り出し範囲
	min_start = max(0, event_idx - max_margin)
	max_start = min(total_samples - window_size, event_idx - min_margin)

	if min_start <= max_start:
		# 通常：イベントが margin 内に収まるようランダムに開始位置を決定
		start_idx = np.random.randint(min_start, max_start + 1)
	else:
		# フォールバック：イベントを含む範囲で通常抽出
		min_start = max(0, event_idx - window_size + 1)
		max_start = min(event_idx, total_samples - window_size)
		if min_start > max_start:
			raise ValueError(f'イベント位置が不正: event_idx={event_idx}')
			print(t)
			print(time)
		start_idx = np.random.randint(min_start, max_start + 1)

	return data[:, start_idx : start_idx + window_size]


silixa_event_file = '/workspace/data/silixa/FORGE_DFIT_NAV.csv'
event_df = pd.read_csv(
	silixa_event_file,
)

event_df['File Timestamp (UTC)'] = pd.to_datetime(event_df['File Timestamp (UTC)'])

event_times = pd.to_datetime(event_df['File Timestamp (UTC)'], utc=True)

# 時刻を抽出する正規表現パターン
pattern = re.compile(r'UTC_(\d{8})_(\d{6})')

preprocessed_dir = '/workspace/data/silixa/raw_78B_npy'
preprocessed_files = np.sort(list(Path(preprocessed_dir).glob('*.npy')))

preprocessed_files_time = []
for path in preprocessed_files:
	m = pattern.search(path.name)
	if not m:
		print('ERROR:No match found:', path.name)
		sys.exit()

	dt_str = m.group(1) + m.group(2)  # '20220417' + '034611'
	# tz をはっきり UTC と指定
	dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
	preprocessed_files_time.append(dt)

preprocessed_files_time = np.array(preprocessed_files_time)
# イベントを含まないTDMSファイルのリスト

resample_dir = Path('/workspace/data/silixa/raw_78B_npy')
extract_events = []
for i, time in enumerate(event_times):
	idx = np.where(
		(preprocessed_files_time < time)
		& (preprocessed_files_time + timedelta(seconds=15) >= time)
	)[0]
	t = pd.Timestamp(preprocessed_files_time[idx][0])
	file = f'FORGE_DFIT_UTC_{t:%Y%m%d_%H%M%S}.202_1kHz.npy'
	data = np.load(resample_dir / file)

	extract_data = extract_2s_window(data, t, time, seed=None)

	if i < 20:
		plt.figure()
		plt.imshow(extract_data, aspect='auto', vmin=-1, vmax=1, cmap='seismic')
		plt.title(time.strftime('%Y-%m-%d %H:%M:%S'))
		plt.savefig(
			f'/workspace/image/extract_event/{time.strftime("%Y%m%d_%H%M%S")}.png'
		)
		plt.cla()
		plt.clf()
		plt.close()
	extract_events.append(extract_data)
# %%
