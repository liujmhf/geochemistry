
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 设置随机种子
torch.manual_seed(42)

# 读取训练数据
# train_data = pd.read_excel('SVM数据.xlsx')
train_data = pd.read_excel('样本数据.xlsx')
features = train_data.iloc[:, 0:12].values
labels = train_data.iloc[:, 12].values

#数据标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
labels = pd.factorize(labels)[0]

#标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

#划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# X_train_tensor = torch.FloatTensor(X_train)
# y_train_tensor = torch.LongTensor(y_train)
# X_test_tensor = torch.FloatTensor(X_test)
# y_test_tensor = torch.LongTensor(y_test)

# 构建神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 增加隐藏层大小
        self.fc2 = nn.Linear(128, 128)
        # 隐藏层到输出层
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 添加Dropout以减少过拟合
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 添加Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 添加Dropout
        x = self.softmax(self.fc3(x))
        return x

# 实例化模型、定义损失函数和优化器
input_size = X_train.shape[1]  # 特征数量
num_classes = len(set(y_train))  # 类别数量
model = NeuralNetwork(input_size, num_classes)
criterion = nn.NLLLoss()  # 使用负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

#训练模型
num_epochs = 1000
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # 设定为训练模式
    optimizer.zero_grad()
    outputs = model(X_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)
    loss.backward()  # 反向传播
    optimizer.step()

    # 计算训练集准确率
    _, train_pred = torch.max(outputs, 1)
    train_accuracy = (train_pred == y_train_tensor).float().mean().item()

    # 记录损失和准确率
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        _, val_pred = torch.max(val_outputs.data, 1)

        val_accuracy = (val_pred == y_val_tensor).float().mean().item()

        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)

    # 每20个epoch打印一次损失和准确率
    if (epoch + 1) % 20 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')


# 验证模型
model.eval()  # 设定为评估模式
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, predicted = torch.max(val_outputs.data, 1)

# 分类报告
print(classification_report(y_val_tensor.numpy(), predicted.numpy()))

# 读取测试数据
test_data = pd.read_excel('SVM待预测样本.xlsx')
X_test = test_data.iloc[:, 0:12].values
X_test = scaler.transform(X_test)  # 标准化测试数据
X_test_tensor = torch.FloatTensor(X_test)
# 转换为Tensor
# X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)

# 预测
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_pred_proba = torch.exp(test_outputs)  # 计算每个类别的概率

test_pred_proba *= 100

predicted_df = pd.DataFrame(test_pred_proba.numpy(), columns=[f'PClass{i}' for i in range(num_classes)])

result_df = pd.concat([test_data, predicted_df], axis=1)

# 保存结果
result_df.to_excel('test_resFnn.xlsx', index=False)
print("测试结果已保存为 'test_resFnn.xlsx'")

# 打印分类报告
if 'true_label' in test_data.columns:
    y_test = test_data['true_label'].values
    y_test_tensor = torch.LongTensor(y_test)
    with torch.no_grad():
        test_outputs = model(torch.FloatTensor(scaler.transform(test_data.iloc[:, 0:12].values)).unsqueeze(1))  # 标准化并预测
        _, test_pred_max = torch.max(test_outputs.data, 1)  # 获取预测的最大概率类别
    print(classification_report(y_test_tensor.numpy(), test_pred_max.numpy()))

# 绘制训练损失和准确率图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

