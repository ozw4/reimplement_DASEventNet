# %%
import datetime as dt
import re
import threading
import time
from pathlib import Path
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
import torch
from build_model import build_model
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

JST = dt.timezone(dt.timedelta(hours=9))  # ★追加

start_dt = dt.datetime(2022, 4, 22, 16, 0, 0, tzinfo=JST)
end_dt = dt.datetime(2022, 4, 23, 18, 0, 0, tzinfo=JST)


def window_generator(
	file_queue, fs: int, win_sec: int = 2, overlap: float = 0.0, n_ch: int = 1021
):
	assert 0 <= overlap < 1
	step = int(win_sec * (1 - overlap) * fs)  # サンプル数
	step_sec = step / fs  # 秒数
	buf = np.empty((n_ch, 0), dtype=np.float32)
	current_ts = None  # ← 追加

	while True:
		# ---------- fill buffer ----------
		while buf.shape[1] < win_sec * fs:
			ts, path, gap_sec = file_queue.pop_next(block=True)
			if current_ts is None:
				current_ts = ts  # 初回のみ基準時刻をセット
			if gap_sec > 0:
				missing = np.zeros((n_ch, int(gap_sec * fs)), dtype=np.float32)
				buf = np.hstack([buf, missing])
			data = np.load(path, mmap_mode='r')
			buf = np.hstack([buf, data])

		# ---------- yield ----------
		win = buf[:, : win_sec * fs]
		yield win, current_ts  # ← (窓, UTC 時刻)

		# ---------- advance ----------
		buf = buf[:, step:]
		current_ts += dt.timedelta(seconds=step_sec)


# ---------- FileQueue ----------
class FileQueue(PriorityQueue):
	"""時刻をキーに Path を保持する優先度付きキュー。"""

	ts_pat = re.compile(r'UTC_(\d{8})_(\d{6})')
	SEC_PER_FILE = 15

	def _ts(self, path: Path) -> dt.datetime:
		m = self.ts_pat.search(path.name)
		if not m:
			raise ValueError(f'Timestamp not found: {path.name}')
		ymd, hms = m.groups()
		# ① タイムゾーン付き datetime を返す
		return dt.datetime.strptime(ymd + hms, '%Y%m%d%H%M%S').replace(tzinfo=JST)

	def push(
		self,
		path: Path,
		start_dt: dt.datetime | None = None,
		end_dt: dt.datetime | None = None,
	):
		ts = self._ts(path)

		# 時刻フィルタ
		if start_dt and ts < start_dt:
			return
		if end_dt and ts >= end_dt:
			return

		self.put((ts, path))

	def pop_next(self, block=True, timeout=None):
		"""* キュー先頭を pop し Path を返す
		* 直前ファイルの想定終了時刻との差分 (gap_sec) も返す
		"""
		ts, path = self.get(block, timeout)  # ★ここで pop される
		gap_sec = 0

		# 直前時刻との差分を計算
		if hasattr(self, '_prev_ts'):
			expected = self._prev_ts + dt.timedelta(seconds=self.SEC_PER_FILE)
			gap_sec = (ts - expected).total_seconds()
			if gap_sec < 0:  # 理論上は負にならないはずだが念のため
				gap_sec = 0

		# 次回比較用に保持
		self._prev_ts = ts
		return ts, path, gap_sec


file_queue = FileQueue()


# ---------- 1) 既存ファイルをスキャン ----------
def scan_existing(data_dir: Path):
	for p in sorted(data_dir.glob('*.npy')):
		file_queue.push(p, start_dt, end_dt)
	print(f'[DirScanner] queued {file_queue.qsize()} existing files')


data_dir = Path('/workspace/data/silixa/raw_78B_npy')
file_queue = FileQueue()

scan_existing(data_dir)  # 起動時一発呼び出し

TS_RE = re.compile(r'UTC_(\d{8})_(\d{6})')


def ts_from_name(name: str) -> dt.datetime:
	ymd, hms = TS_RE.search(name).groups()
	return dt.datetime.strptime(ymd + hms, '%Y%m%d%H%M%S')


# ---------- 2) 新規ファイルを監視 ----------
class AddEvent(FileSystemEventHandler):
	def on_created(self, event):
		p = Path(event.src_path)
		if p.suffix == '.npy':
			file_queue.push(p, start_dt, end_dt)
			print(f'[FileWatch] new file queued: {p.name}')


observer = Observer()
observer.schedule(AddEvent(), path=str(data_dir), recursive=False)
observer.start()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model().to(device)

model_dir = Path('/workspace/output/train')
model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
model.eval()  # 評価モードに設定


plt.figure()


# ---------- 3) 連続処理スレッド（例） ----------
def consume_files():
	fs = 1000
	win_gen = window_generator(file_queue, fs, win_sec=2, overlap=0.0, n_ch=1021)

	pbar = tqdm(total=None, bar_format='{l_bar}{bar}| {n_fmt} win  {postfix}')
	for win, ts_jst in win_gen:
		# --- 進捗バー用の時刻 (JST) ---
		ts_str = ts_jst.strftime('%Y-%m-%d %H:%M:%S')  # ここでそのまま文字列化
		pbar.set_postfix_str(ts_str)
		pbar.update()

		# --- 前処理 & 推論 ---
		win = (win - win.mean(axis=1, keepdims=True)) / win.std(axis=1, keepdims=True)
		win_tensor = (
			torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
		)
		with torch.no_grad():
			pred = model(win_tensor)
		if torch.sigmoid(pred).item() > 0.5:
			plt.imshow(win, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
			plt.show()
			time.sleep(1)


worker = threading.Thread(target=consume_files, daemon=True)
worker.start()

# ---------- メインスレッドは適宜待機 ----------
try:
	while True:
		time.sleep(1)
except KeyboardInterrupt:
	observer.stop()
observer.join()


# %%
