# %%
import pathlib as Path

import numpy as np
from nptdms import TdmsFile
from scipy.signal import resample_poly

from proc.util.filters import bandpass_filter

data_dir = Path.Path('/workspace/data/silixa')

nptdms_files = list(data_dir.glob('*.tdms'))
np.sort(nptdms_files)

for nptdms_file in nptdms_files:
	# check already resampled file
	if Path.Path.exists(data_dir / 'raw_78B_npy' / (nptdms_file.stem + '_1kHz.npy')):
		# print(f'File {nptdms_file.name} already processed.')
		continue
	print(f'Processing file: {nptdms_file.name}')
	try:
		tdms_file = TdmsFile.read(nptdms_file)
	except Exception as e:
		print(f'Error reading {nptdms_file.name}: {e}')
		continue
	# 'Measurement' グループ内のすべてのチャンネル名を取得
	group = tdms_file['Measurement']
	channel_names = group.channels()

	# 各チャンネルをNumPy配列としてまとめて取得（辞書形式 or 配列形式）
	channel_data_dict = {ch.name: ch.data for ch in channel_names}

	seis = np.vstack([ch.data for ch in channel_names])
	# print(seis.shape)

	# extract using silixa processed sgy info
	seis78A = seis[69:1079]  # CH 70:1079
	seis78B = seis[1195:2401]  # CH 1196:2401

	# extract 1021 CH (Yu et al., 2024)  remove noisy trace
	seis78B = seis78B[168:-17]  # CH 1364:2384

	# resample 4kHz to 1kHz
	downsampled_seis78B = resample_poly(
		seis78B.astype(np.float32), up=1, down=4, axis=1
	)

	filtered_seis78B = bandpass_filter(downsampled_seis78B, fs=1000)
	median_per_sample = np.median(filtered_seis78B, axis=0)  # shape = (n_samples,)
	denoised_seis78B = filtered_seis78B - median_per_sample

	savename = str(nptdms_file.stem) + '_1kHz.npy'
	np.save(data_dir / 'raw_78B_npy' / savename, denoised_seis78B)
	# sys.exit()
# %%
