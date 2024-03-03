# Trilateration Estimator
[3辺測量](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_IROS_2009/papers/0978.pdf)のpython実装
1. アンカー(計測器)とビークル(測定対象)の間のノイズ付きの距離を測る
1. 複数の測定値から二次計画法を用いてアンカーからの距離を推定する
1. 推定した各アンカーからの距離からビークルの座標を推定する

## ファイル構成
- main.py : シミュレーションを実行
- anchor.py : 距離を計測する機器(アンカー)クラスを実装
- vehicle.py : 移動ロボットクラスを実装
- trilateration.py : 三辺測量クラスを実装
- sampling.py : 位置推定の分散共分散行列を計算
