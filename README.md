# codex_sample

Pythonで動作する **9x9の4目並べ（重力なし）** のサンプルです。

## 追加ファイル
- `connect4_dqn.py`
  - 人間 vs コンピュータで対戦可能
  - DQNを用いた自己対戦学習に対応

## 使い方

### 1. 学習（自己対戦）
```bash
python connect4_dqn.py train --episodes 3000 --save model.pth
```

主なオプション:
- `--episodes` 学習エピソード数
- `--batch-size` ミニバッチサイズ
- `--target-update` ターゲットネットワーク更新間隔
- `--device` `cpu` / `cuda`

### 2. 対戦（人間 vs 学習済みモデル）
```bash
python connect4_dqn.py play --model model.pth
```

- 人間は `X`、AIは `O` です。
- 入力はマス座標（`row col`、それぞれ `0`〜`8`）。

## 依存ライブラリ
- `numpy`
- `torch`

必要なら以下で導入:
```bash
pip install numpy torch
```
