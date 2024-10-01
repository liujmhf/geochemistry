
# 模型参数调整：调整 SVM 的超参数，如 C 和 gamma。
# 调整 SVM 的超参数。

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 0:12].values  # 特征值
    y = df.iloc[:, 12].values
    return X, y

# 训练模型
def train_model(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 使用网格搜索来优化超参数
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf'],
    }

    grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f'最佳参数: {grid_search.best_params_}')
    model = grid_search.best_estimator_

    # 保存模型和标准化器
    torch.save(model, 'svm_model.pth')
    torch.save(scaler, 'scaler.pth')

    return model, scaler, X_test, y_test

# 测试模型
def test_model(model, scaler, test_file_path):
    # 加载测试数据
    test_df = pd.read_excel(test_file_path)
    X_test = test_df.iloc[:, 0:12].values  # 使用与训练相同的特征列
    X_test_scaled = scaler.transform(X_test)  # 标准化测试数据

    # 进行预测
    probabilities = model.predict_proba(X_test_scaled)
    return probabilities

# 绘制损失图
def plot_losses(cv_scores):
    plt.plot(cv_scores, marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # 训练数据文件路径
    train_file_path = '样本数据.xlsx'
    X, y = load_data(train_file_path)

    # 训练模型
    model, scaler, X_test, y_test = train_model(X, y)

    # 测试数据文件路径
    test_file_path = 'SVM待预测样本.xlsx'
    probabilities = test_model(model, scaler, test_file_path)

    # 创建 DataFrame 保存测试结果
    results_df = pd.DataFrame(probabilities, columns=['C0P', 'C1P'])

    original_test_df = pd.read_excel(test_file_path)
    output_df = pd.concat([original_test_df, results_df], axis=1)

    # 保存合并结果到Excel文件
    output_df.to_excel('test_svm.xlsx', index=False)

    # 打印训练精度
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'模型准确率: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))


