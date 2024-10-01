
# 增加树的数量、 限制树的深度以及使得模型在训练时自动根据类别数量调整权重
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 读取训练数据
train_data = pd.read_excel('样本数据.xlsx')

# 划分特征和标签
X = train_data.iloc[:, 0:12].values  # 特征值
y = train_data.iloc[:, 12].values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器，处理不平衡数据
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')

# 训练模型
rf_model.fit(X_train, y_train)

# 验证模型
y_val_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# 保存模型
joblib.dump(rf_model, 'forest_model.pkl')

# 读取测试数据
test_data = pd.read_excel('SVM待预测样本.xlsx')

# 提取特征
X_test = test_data.iloc[:, 0:12].values

# 进行概率预测
probabilities = rf_model.predict_proba(X_test)

# 将概率乘以100并转换为命名列
probabilities_df = pd.DataFrame(probabilities * 100, columns=[f'Prob{i}' for i in range(probabilities.shape[1])])

# 将概率添加到测试数据中
test_data = pd.concat([test_data, probabilities_df], axis=1)

# 保存测试结果
test_data.to_excel('test_resultsRF.xlsx', index=False)

print("测试结果已保存到 'test_resultsRF.xlsx'")