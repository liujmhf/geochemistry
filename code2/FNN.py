
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
# data
data = pd.read_excel('SVM数据.xlsx')
# 数据预处理
features = data.iloc[:, 0:12].values  # 获取特征数据
labels = data.iloc[:, 12].values

# 将标签转换为整数编码
labels = pd.factorize(labels)[0]

# 标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

#  定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层2
        self.fc3 = nn.Linear(64, num_classes)  # 输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型、定义损失函数和优化器
input_size = X_train.shape[1]
num_classes = len(set(labels))
model = NeuralNetwork(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 训练模式
    optimizer.zero_grad()
    outputs = model(X_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)
    loss.backward()  # 反向传播
    optimizer.step()

    if (epoch+1) % 10 == 0:  # 每10个epoch输出一次损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 评估模式
with torch.no_grad():
    predicted = model(X_test_tensor)
    _, predicted_classes = torch.max(predicted, 1)
    accuracy = (predicted_classes == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')
