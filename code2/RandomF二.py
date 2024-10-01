import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib  # 用于保存模型

# 读取数据
data = pd.read_excel('样本数据.xlsx')

# 分离特征和标签
X = data.iloc[:, 0:12].values  # 特征值
y = data.iloc[:, 12].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_clf = RandomForestClassifier(random_state=42)

# 定义超参数范围
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],     # 叶子节点的最小样本数
}

# 使用网格搜索优化超参数
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid,
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 使用最佳模型进行预测
best_rf_clf = grid_search.best_estimator_
y_pred = best_rf_clf.predict(X_test)

# 计算训练精度
accuracy = accuracy_score(y_test, y_pred)
print(f'最佳模型训练精度: {accuracy:.2f}')

# 保存模型
joblib.dump(best_rf_clf, 'forest_model2.pkl')

# 可视化训练过程中的损失和精度
train_accuracy = []
for i in range(1, 301):  # 增加树的数量范围
    best_rf_clf.set_params(n_estimators=i)  # 逐步增加树的数量
    best_rf_clf.fit(X_train, y_train)
    train_accuracy.append(accuracy_score(y_train, best_rf_clf.predict(X_train)))

# 绘制精度图
plt.plot(range(1, 301), train_accuracy, label='训练集精度')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('训练精度与树的数量的关系')
plt.legend()
plt.show()

# 测试模型并读取新的测试文件
test_data = pd.read_excel('SVM待预测样本.xlsx')
X_test_final = test_data.iloc[:, 0:12].values  # 读取特征值

# 计算分类概率
proba = best_rf_clf.predict_proba(X_test_final)
proba=proba*100
# 将预测结果和概率合并到测试数据中
test_results = pd.DataFrame(X_test_final, columns=test_data.columns[0:12])
test_results['pro'] = best_rf_clf.predict(X_test_final)
test_results['Prob1'] = proba[:, 0]
test_results['Prob2'] = proba[:, 1]

# 将结果保存到Excel文件中
test_results.to_excel('test_resultsRF2.xlsx', index=False)
print("测试结果已保存到 test_resultsRF2.xlsx")
