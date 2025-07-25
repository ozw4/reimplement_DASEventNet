# %%
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from build_model import build_model

from pathlib import Path
from queue import PriorityQueue
import threading, re, time, datetime as dt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



def window_generator(file_queue, fs: int, win_sec: int = 2, overlap: float = 0.0):
    step = int(win_sec * (1 - overlap) * fs)
    buf = np.empty((n_ch, 0), dtype=np.float32)  # 残りデータ保持

    while True:
        # buffer が足りなければ次ファイルを読む
        while buf.shape[1] < win_sec * fs:
            path, gap_sec = file_queue.pop_next(block=True)  # block=True: 無ければ待つ
            if gap_sec > 0:                      # 欠損 → ゼロ埋め
                missing = np.zeros((n_ch, int(gap_sec * fs)), dtype=np.float32)
                buf = np.hstack([buf, missing])
            if path is not None:
                data = np.load(path, mmap_mode='r')
                buf = np.hstack([buf, data])

        # ウィンドウ切り出し
        win = buf[:, :win_sec * fs]
        yield win

        # バッファを前進
        buf = buf[:, step:]

data_dir = Path('/workspace/data/silixa/raw_78B_npy')
files = np.sort(list(data_dir.glob('*.npy')))

pat = re.compile(r'UTC_(\d{8})_(\d{6})')
utc_times = []  # datetime 型でほしい場合

for p in files:
	m = pat.search(p.name)
	if m:
		ymd, hms = m.groups()  # ('20220418', '163156')
		utc_times.append(datetime.strptime(ymd + hms, '%Y%m%d%H%M%S'))
utc_times = np.array(utc_times)

current_process_time = min(utc_times)  # 最も古い UTC 時刻を取得
delta_min = 60
process_files_idx = (utc_times >= current_process_time) & (
	utc_times < (current_process_time + timedelta(minutes=delta_min))
)
process_files = files[process_files_idx]

n_ch = 1021

for p, t in zip(process_files, utc_times[process_files_idx]):



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model().to(device)

model_dir = Path('/workspace/output/train')
model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
