# reimplement_DASEventNet

このリポジトリは、Distributed Acoustic Sensing(DAS)データを用いた微小地震検出モデル「DASEventNet」を再実装するための実験コードをまとめたものです。

## 構成
- `Pengliang-Yu-DASEventNet-7282648` : Pengliang Yu氏が公開したDASEventNet実装を含むディレクトリ。
- `proc/` : TDMS形式のデータを前処理するスクリプト群。
- `notebook/DASEventNet_Preprocessing_Example.ipynb` : 前処理の手順例を示すJupyterノートブック。
- `data/` : 予測カタログ`prediction_catalog_final_loc.dat`やSilixa社のDASデータ取得スクリプトを収録。

本リポジトリは研究用途の再現実装を目的としており、完全な学習パイプラインやテスト環境は整備されていません。FORGEプロジェクトで公開されているデータセットを用いた実験や、元実装との比較等にご活用ください。
