import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import yfinance as yf
import mplfinance as mpf
import torch
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
from sklearn.preprocessing import MinMaxScaler

# 評価関数RMSE
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))





#ターゲットを指定
ticker = "AAPL"


#データを収集
data = yf.download(ticker, start= "2010-01-01")
df = data

#ローソク足グラフの表示
# mpf.plot(df, type="candle",volume=True,figratio=(10,5))

#５日間の終値の移動平均を指標””sma5に加える
price = df["Close"]
span = 5
print(price)
#５に満たない部分は現在までの日数分の移動平均
df["sma05"] = price.rolling(window=span, min_periods=1).mean()
# print(df["sma05"])

#index値を落としている
df = df['sma05'].values
print(df)

#株価df1次元配列に対してreshape(-1, 1)とすると、その配列を要素とする2次元1列の配列にする
df = df.reshape(-1,1)
df = df.astype("float32") 

# 0から１に正規化
scaler = MinMaxScaler(feature_range = (0, 1))
df_scaled = scaler.fit_transform(df)
print(df_scaled)
#正規化された株価の推移を図示して確認する
# plt.plot(df_scaled)
plt.xlabel("time point(day)")
plt.ylabel("stock price($)")
plt.title(f"{ticker}'s stock price")
# plt.show()


#dataset作成

train_size = int(len(df_scaled) * 0.80) #学習サイズ(全範囲の8割)
test_size = len(df_scaled) - train_size #全データから学習サイズを引けばテストサイズになる(3割)
train = df_scaled[0:train_size,:] #全データから学習の個所を抜粋
test = df_scaled[train_size:len(df_scaled),:] #全データからテストの個所を抜粋
print("train size: {}, test size: {} ".format(len(train), len(test)))

time_stemp = 10 #今回は10個のシーケンシャルデータを1固まりとするので10を設定
n_sample = train_size - time_stemp + 1 #学習予測サンプルはt=10~nなので

#シーケンシャルデータの固まり数、シーケンシャルデータの長さ、RNN_cellへの入力次元(1次元)に形を成形
input_data = np.zeros((n_sample, time_stemp, 1)) #シーケンシャルデータを格納する箱を用意(入力)
correct_input_data = np.zeros((n_sample, 1)) #シーケンシャルデータを格納する箱を用意(正解)

print(input_data.shape)
print(correct_input_data.shape)

for i in range(n_sample):
    input_data[i] = df_scaled[i:i+time_stemp].reshape(-1, 1)
    correct_input_data[i] = df_scaled[i+time_stemp:i+time_stemp+1]

input_data = torch.tensor(input_data, dtype=torch.float) #Tensor化(入力)
correct_data = torch.tensor(correct_input_data, dtype=torch.float) #Tensor化(正解)
dataset = torch.utils.data.TensorDataset(input_data, correct_data) #データセット作成
train_loader = DataLoader(dataset, batch_size=4, shuffle=True) #データローダー作成

class My_rnn_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_rnn_net, self).__init__()

        self.input_size = input_size #入力データ(x)
        self.hidden_dim = hidden_dim #隠れ層データ(hidden)
        self.n_layers = n_layers #RNNを「上方向に」何層重ねるか？の設定 ※横方向ではない
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size) #全結合層でhiddenからの出力を1個にする

    def forward(self, x):
        #h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        #y_rnn, h = self.rnn(x, h0)
        y_rnn, h = self.rnn(x, None) #hidden部分はコメントアウトした↑2行と同じ意味になっている。
        y = self.fc(y_rnn[:, -1, :]) #最後の時刻の出力だけを使用するので「-1」としている

        return y

#RNNの設定
n_inputs  = 1
n_outputs = 1
n_hidden  = 64 #隠れ層(hidden)を64個に設定
n_layers  = 1

net = My_rnn_net(n_inputs, n_outputs, n_hidden, n_layers) #RNNをインスタンス化
print(net) #作成したRNNの層を簡易表示

#おすすめのtorchinfoでさらに見やすく表示
batch_size = 4
summary(net, (batch_size, 10, 1))

loss_fnc = nn.MSELoss() #損失関数はMSE
optimizer = optim.Adam(net.parameters(), lr=0.001) #オプティマイザはAdam
loss_record = [] #lossの推移記録用
device = torch.device("cuda:0" if torch.cuda. is_available() else "cpu")  #デバイス(GPU or CPU)設定 
epochs = 10 #エポック数

net.to(device) #モデルをGPU(CPU)へ

for i in range(epochs+1):
    net.train() #学習モード
    running_loss =0.0 #記録用loss初期化
    for j, (x, t) in enumerate(train_loader): #データローダからバッチ毎に取り出す
        x = x.to(device) #シーケンシャルデータをバッチサイズ分だけGPUへ
        optimizer.zero_grad() #勾配を初期化
        y = net(x) #RNNで予測
        y = y.to('cpu') #予測結果をCPUに戻す
        loss = loss_fnc(y, t) #MSEでloss計算
        loss.backward()  #逆伝番        
        optimizer.step()  #勾配を更新        
        running_loss += loss.item()  #バッチごとのlossを足していく
    running_loss /= j+1 #lossを平均化
    loss_record.append(running_loss) #記録用のlistにlossを加える

    """以下RNNの学習の経過を可視化するコード"""
    if i%1 == 0: #今回は100エポック毎に学習がどう進んだか？を表示させる
        print('Epoch:', i, 'Loss_Train:', running_loss)
        input_train = list(input_data[0].reshape(-1)) #まず最初にt＝0～9をlist化しておく
        predicted_train_plot = [] #学習結果plot用のlist
        net.eval() #予測モード
        for k in range(n_sample): #学習させる点の数だけループ
            x = torch.tensor(input_train[-time_stemp:]) #最新の10個のデータを取り出してTensor化
            x = x.reshape(1, time_stemp, 1) #予測なので当然バッチサイズは1
            x = x.to(device).float() #GPUへ
            y = net(x) #予測
            y = y.to('cpu') #結果をCPUへ戻す
            if k <= n_sample-2: 
                input_train.append(input_data[k+1][9].item())
            predicted_train_plot.append(y[0].item())


# plt.plot(range(len(df_scaled)), df_scaled, label='Correct')
# plt.plot(range(time_stemp, time_stemp+len(predicted_train_plot)), predicted_train_plot, label='Predicted')
# plt.legend()
# plt.show()

#学習の時と同じ感じでまずは空のデータを作る
time_stemp = 10
n_sample_test = len(df_scaled) - train_size #テストサイズは学習で使ってない部分
test_data = np.zeros((n_sample_test, time_stemp, 1))
correct_test_data = np.zeros((n_sample_test, 1))


#t=train_size - time_stemp 以降のデータを抜粋してシーケンシャルデータとして格納していく
start_test = train_size - time_stemp 
for i in range(n_sample_test-1):
    test_data[i] = df_scaled[start_test+i : start_test+i+time_stemp].reshape(-1, 1)
    # print(start_test+i+time_stemp)
    # print(df_scaled[start_test+i+time_stemp : start_test+i+time_stemp+1])
    # print(df_scaled[start_test+i+time_stemp : start_test+i+time_stemp+1].size)
    # print(df_scaled[start_test+i+time_stemp : start_test+i+time_stemp+1].shape)
 
    correct_test_data[i] = df_scaled[start_test+i+time_stemp : start_test+i+time_stemp+1]


#以下は学習と同じ要領
input_test = list(test_data[0].reshape(-1))
predicted_test_plot = []
net.eval()
for k in range(n_sample_test):
    x = torch.tensor(input_test[-time_stemp:])
    x = x.reshape(1, time_stemp, 1)
    x = x.to(device).float()
    y = net(x)
    y = y.to('cpu')
    if k <= n_sample_test-2: 
         input_test.append(test_data[k+1][9].item())
    predicted_test_plot.append(y[0].item())

predicted_test_plot = scaler.inverse_transform(pd.DataFrame(predicted_test_plot))
test = scaler.inverse_transform(pd.DataFrame(test))


from sklearn.metrics import mean_squared_error, mean_absolute_error
# MSEとMAEを計算する
mse = mean_squared_error(test, predicted_test_plot)
mae = mean_absolute_error(test, predicted_test_plot)

print("MSE: {:.4f}".format(mse))
print("MAE: {:.4f}".format(mae))

plt.plot(range(len(test)), test, label='Correct')
plt.plot(range(len(predicted_test_plot)-1), pd.DataFrame(predicted_test_plot)[:-1], label='Predicted')
plt.legend()
plt.show()

