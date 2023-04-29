import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 株価データの取得
data = yf.download("AAPL", start="2010-01-01")

# データの前処理
scaler = MinMaxScaler()
data["Close"] = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# データセットの作成
look_back = 30
look_go = 64
def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back)])
        y.append(dataset[i+look_back])
    return np.array(X), np.array(y)

train_size = int(len(data) * 0.8)
train_data = data["Close"][:train_size]
test_data = data["Close"][train_size:]
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Transformerのモデルの定義
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model*look_back, output_dim)
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
    
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# ハイパーパラメータの設定
num_epochs = 30
batch_size = 64
lr = 0.001
d_model = 64
nhead = 8
num_layers = 2

# データセットとデータローダーの定義
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの学習
model = TransformerModel(input_dim=1, output_dim=1, d_model=d_model, nhead=nhead, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2).expand(-1, -1, 64))
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# モデルの予測を行う
model.eval()
with torch.no_grad():
  inputs = torch.from_numpy(X_test).float().unsqueeze(2).expand(-1, -1, 64)
  outputs = model(inputs)
  predicted = scaler.inverse_transform(outputs.numpy())

y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
mse = nn.MSELoss()(torch.from_numpy(predicted), torch.from_numpy(y_test_inverse)).item()
print('Test MSE: {:.4f}'.format(mse))

# テストデータのプロット
import matplotlib.pyplot as plt
plt.plot(y_test_inverse, label='Actual')
plt.plot(predicted, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

