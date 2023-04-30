# stock_price_prediction



# 株価予測モデル

このプロジェクトは、PyTorchを使用して株価の時系列データを予測するモデルを構築することを目的としています。

## 動作環境

以下の環境で動作確認済みです。

- Python 3.7.6
- PyTorch 1.9.0
- numpy 1.19.2
- pandas 1.1.3
- matplotlib 3.3.2
- scikit-learn 0.24.1

## データセット

データセットは、Yahoo FinanceからダウンロードしたAAPL（Apple Inc.）の時系列株価データを使用しています。データは、以下の特徴量から構成されています。

- Date: 日付
- Open: 初値
- High: 高値
- Low: 安値
- Close: 終値
- Adj Close: 調整後終値
- Volume: 出来高

## モデル

本プロジェクトでは、LSTM（Long Short-Term Memory）ネットワークを使用して株価の予測を行いました。LSTMは、時系列データの学習に適したニューラルネットワークであり、長期的な依存関係を考慮することができます。

LSTMのアーキテクチャは、以下のようになっています。

```
LSTM(
  (lstm): LSTM(1, 64, num_layers=2, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)
```

入力は、1つの特徴量（終値）のみです。LSTMの出力は、64次元のベクトルであり、全結合層を介して1つの値に変換されます。

## 学習

モデルは、以下のような手順で学習されます。

1. データをトレーニングセットとテストセットに分割します。
2. トレーニングセットを使用してモデルをトレーニングします。
3. テストセットを使用してモデルの予測精度を評価します。

学習時には、以下のハイパーパラメータが設定されます。

- `num_epochs`: エポック数
- `learning_rate`: 学習率
- `input_size`: 入力サイズ（特徴量の数）
- `hidden_size`: 隠れ層のサイズ
- `num_layers`: LSTMの層数
- `batch_size`: バッチサイズ

## 評価
平均二乗誤差（MSE）: 予測と実際の株価の差を二乗して平均したもの。
平均絶対誤差（MAE）: 予測と実際の株価の差の絶対値を平均したもの。
