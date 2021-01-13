import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import mean_squared_error
import sys

class Model:
    def __init__(self, criterion, max_depth):
        # 模型初始化
        self.dtc = DTC(criterion=criterion, max_depth=max_depth)

    def train(self, X, y):
        #模型构建
        self.dtc.fit(X, y)

    def predict(self, X):
        pred_y = self.dtc.predict(X)
        return pred_y


def age_handler(ages):
    # 将年龄分箱
    age_min = ages.min() // 10 * 10
    age_max = ages.max() // 10 * 10 + 10
    bins = []
    for i in range(age_min, age_max + 1, 10):
        bins.append(i)
    ages = pd.cut(ages, bins, labels=False)
    return ages


def sex_handler(sex):
    # 将性别转换为整型
    sexs = sex.unique()
    sex_mapping = {sexs[i]: i for i in range(sexs.shape[0])}
    sex = pd.to_numeric(sex.map(sex_mapping)).astype(int)
    return sex


def nan_handler(data):
    """
    先丢弃缺失值过多的列和行，然后补充缺失值
    由于对大量人群的体检可看做随机抽样过程，因此假设所有数据服从正态分布
    """
    # 缺失值超过thresh_limit则丢弃
    # 列
    cols = ["乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝e抗体","乙肝核心抗体"]
    data.drop(columns=cols , inplace=True)
    # 行
    col_len = data.shape[1]
    rate = 0.7  # 阈值
    thresh_limit = col_len * rate
    data.dropna(axis=0, thresh=thresh_limit, inplace=True)

    # 对剩下的缺失值进行填充
    cols = data.columns
    for col in cols:
        data[col] = data[col].fillna(data[col].mean())

    return data

def preprocess(data):
    # 日期不影响预测，丢掉
    data = data.drop(columns='体检日期')
    # 去除重复元素
    data = data.drop_duplicates()
    # 处理年龄和性别
    data['年龄'] = age_handler(data['年龄'])
    data['性别'] = sex_handler(data['性别'])

    # 处理缺失值
    data = nan_handler(data)
    return data

def div_xy(data):
    data = data.iloc[:, 1:]
    cols = list(data.columns.values)
    cols.remove('血糖')
    X = data[cols].astype('str')
    y = data[['血糖']].astype('str')
    return X,y

if __name__ == '__main__':
    #模型参数
    cri = 'entropy'
    depth = 5
    model = Model(criterion=cri, max_depth=depth)

    ##读取原始数据并预处理
    test_file = sys.argv[1]
    test_data = pd.read_csv(test_file, encoding='gbk')
    test_data = preprocess(test_data)

    train_data = pd.read_csv(r"d_train.csv", encoding='gbk')
    train_data = preprocess(train_data)
    #数据读取与划分
    X_train,y_train = div_xy(train_data)
    X_test,y_test = div_xy(test_data)
    model.train(X_train, y_train)
    pred_y = model.predict(X_test)

    # number = X.shape[0]
    # div = int(0.8 * number)
    # X_train, y_train = X[:div], y[:div]
    # X_test, y_test = X[div:], y[div:]
    # #模型训练与预测
    # model.train(X_train, y_train)
    # pred_y = model.predict(X_test)
    # # print(model.dtc.score(X_test,y_test))
    # 计算损失函数
    y_test = np.array(y_test['血糖']).astype('float')
    pred_y = pred_y.astype('float')
    loss = sum((pred_y - y_test) ** 2)
    # 最后需要输出预测的准确率或者均方误差等指标值
    f = loss / (X_test.shape[0])
    print("均方误差：",f)
