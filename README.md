# stock_price_prediction

このプロジェクトは、PyTorchを使用して株価の時系列データを予測するモデルを構築することを目的としています。

# 株価予測モデル


![lstm](https://user-images.githubusercontent.com/132123636/235346784-8fad7e7c-fd59-4897-a472-9c1a7f369cb7.png)
## 動作環境

以下の環境で動作確認済みです。

- Python 3.7.16
- torch 1.13.1
- numpy 1.21.6
- pandas 1.3.5
- matplotlib 3.5.3
- scikit-learn 1.0.2

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

本プロジェクトでは、RNN(Recurrent Neural Network),LSTM（Long Short-Term Memory),Transformerの3つのネットワークを使用してそれぞれ株価の予測を行いました。


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
- `num_layers`: モデルの層数
- `batch_size`: バッチサイズ

## 評価
- 平均二乗誤差（MSE）: 予測と実際の株価の差を二乗して平均したもの。
- 平均絶対誤差（MAE）: 予測と実際の株価の差の絶対値を平均したもの。
