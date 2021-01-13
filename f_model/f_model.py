import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import sys

class Model:
    def __init__(self, criterion, max_depth):
        # 模型初始化
        self.dtc = RandomForestClassifier(warm_start=True, random_state=1, n_estimators=26,
                               max_depth=5, max_features='sqrt', min_samples_split=4, min_samples_leaf=2)

    def train(self, X, y):
        # 模型构建
        y = y.values.ravel()
        self.dtc.fit(X, y)
        return model

    def predict(self, X):
        pred_y = self.dtc.predict(X)
        return pred_y


def age_handler(ages):
    # 将年龄分箱
    age_min = int(ages.min() // 10 * 10)
    age_max = int(ages.max() // 10 * 10 + 10)
    bins = []
    for i in range(age_min, age_max + 1, 10):
        bins.append(i)
    ages = pd.cut(ages, bins, labels=False)
    return ages

def nan_handler(data):
    """
    先丢弃缺失值过多的列和行，然后补充缺失值
    由于对大量人群的体检可看做随机抽样过程，因此假设所有数据服从正态分布
    """
    # 缺失值超过thresh_limit则丢弃
    # 列
    row_len = data.shape[0]
    rate = 0.8  # 阈值
    thresh_limit = row_len * rate
    data.dropna(axis=1, thresh=thresh_limit, inplace=True)

    # 对剩下的缺失值进行填充
    cols = data.columns
    for col in cols:
        data[col] = data[col].fillna(data[col].mean())

    return data


def preprocess(data):
    # 去除重复元素
    data = data.drop_duplicates()
    # 处理年龄
    data['年龄'] = age_handler(data['年龄'])

    # 处理缺失值
    data = nan_handler(data)

    return data

def div_xy(data):
    data = data.iloc[:, 1:]
    cols = list(data.columns.values)
    cols.remove('label')
    X = data[cols].astype('str')
    y = data[['label']].astype('str')
    return X,y

if __name__ == '__main__':
    # 模型参数
    cri = 'entropy'
    depth = 5
    model = Model(criterion=cri, max_depth=depth)

    ##读取原始数据并预处理
    test_file = sys.argv[1]
    test_data = pd.read_csv(test_file, encoding='gbk')
    test_data = preprocess(test_data)

    train_data = pd.read_csv(r"f_train.csv", encoding='gbk')
    train_data = preprocess(train_data)
    # 数据读取与划分
    X_train, y_train = div_xy(train_data)
    X_test, y_test = div_xy(test_data)
    model.train(X_train, y_train)
    pred_y = model.predict(X_test)

    #读取原始数据并预处理
    print("准确度：",model.dtc.score(X_test, y_test))
    # 计算损失函数
    y_test = np.array(y_test['label']).astype('float')
    pred_y = pred_y.astype('float')
    # loss = sum((pred_y - y_test) ** 2)
    # # 最后需要输出预测的准确率或者均方误差等指标值
    # f = loss / (2 * (number - div))
    f = mean_squared_error(pred_y, y_test)
    print("均方误差：", f)

    y_len = len(pred_y)
    pos = 0
    pos_cor = 0
    pos_ori = 0
    for i in range(y_len):
        if pred_y[i] == 1:
            pos += 1
            if pred_y[i] == y_test[i]:
                pos_cor += 1
        if y_test[i] == 1:
            pos_ori += 1

    P = pos_cor / pos
    R = pos_cor / pos_ori
    f1 = 2*P*R/(P+R)
    print("f1:",f1)
