import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 株価データを取得する
symbol = "AAPL"
data = yf.download(symbol, start="2010-01-01")

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# 訓練データとテストデータに分割する
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

# LSTMに入力するデータを作成する
def create_dataset(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# PyTorchのデータセットを作成する
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

train_dataset = StockDataset(X_train, Y_train)
test_dataset = StockDataset(X_test, Y_test)

# LSTMのモデルを構築する
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1
hidden_dim = 50
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, output_dim)

# モデルの学習を行う
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
num_epochs = 1
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# モデルの予測を行う
model.eval()
with torch.no_grad():
    inputs = torch.from_numpy(X_test).float().unsqueeze(2)
    outputs = model(inputs)
    predicted_stock_price = scaler.inverse_transform(outputs.numpy())

from sklearn.metrics import mean_squared_error, mean_absolute_error
# MSEとMAEを計算する
mse = mean_squared_error(Y_test, predicted_stock_price)
mae = mean_absolute_error(Y_test, predicted_stock_price)

print("MSE: {:.4f}".format(mse))
print("MAE: {:.4f}".format(mae))

import matplotlib.pyplot as plt
plt.plot(data.index[train_size+look_back:], data["Close"][train_size+look_back:], label="True Stock Price")
plt.plot(data.index[train_size+look_back:], predicted_stock_price, label="Predicted Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()