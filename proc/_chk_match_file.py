import re
from pathlib import Path

# === 設定 ===
txt_path = Path(
	'~/Desktop/daseventnet/data/silixa/get_silixa_raw_tdms_april_2022.txt'
).expanduser()
npy_dir = Path('~/Desktop/daseventnet/data/silixa/raw_78B_npy').expanduser()

# === .tdms 側: ベース名を収集 ===
tdms_bases: set[str] = set()
with txt_path.open() as f:
	for line in f:
		line = line.strip()
		if not line:  # 空行スキップ
			continue
		fname = Path(line).name  # URL なら末尾のファイル名だけ取り出す
		base = re.sub(r'\.tdms$', '', fname)
		tdms_bases.add(base)

# === .npy 側: ベース名を収集 ===
npy_bases: set[str] = set()
for p in npy_dir.glob('*.npy'):
	base = re.sub(r'_1kHz\.npy$', '', p.name)
	npy_bases.add(base)

# === 差分を計算 ===
missing_npy = tdms_bases - npy_bases  # txt にあるが .npy が無い
orphaned_npy = npy_bases - tdms_bases  # .npy だけ存在し txt に無い

# === 結果表示 ===
print(f'◆ txt にあって .npy が無いもの: {len(missing_npy)} 件')
for b in sorted(missing_npy):
	print('  ', b)

print(f'\n◆ .npy だけ存在するもの: {len(orphaned_npy)} 件')
for b in sorted(orphaned_npy):
	print('  ', b)
