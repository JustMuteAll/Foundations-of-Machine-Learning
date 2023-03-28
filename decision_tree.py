# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

params = []
class TreeNode(object):
    
    def __init__(self, model=None, C=None, left=None, right=None):
        self.model = model
        self.C = C
        self.left = left
        self.right = right

def trainLinear(linear, x, y):
    #使用sklearn库的最小二乘估计训练一个线性模型
    linear.fit(x, y)
    return linear

def binaryTrainSet(linear, x, y):
    #根据线性回归模型二分数据集
    #对样本x[i],其线性模型预测值若小于等于0,分到x0集合;若大于0,分到x1集合;相应的标签也划分的y0,y1集合
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    p = linear.predict(x)
    for i in range(p.shape[0]):
        if p[i] <= 0:
            x0.append(x[i])
            y0.append(y[i])
        else:
            x1.append(x[i])
            y1.append(y[i])
    return np.array(x0), np.array(x1), np.array(y0), np.array(y1)

def score(linear, x, y):
    #计算线性模型linear的精度
    right = 0
    p = linear.predict(x)
    for i in range(p.shape[0]):
        if p[i]<=0 and y[i]==-1 or p[i]>0 and y[i]==1:
            right += 1
    return right / x.shape[0]
    
def treeGenerate(root, x, y, precision):
    #递归建造决策树
    root.model = LinearRegression()
    root.model = trainLinear(root.model, x, y)
    params.append([*root.model.coef_, root.model.intercept_])
    x0, x1, y0, y1 = binaryTrainSet(root.model, x, y)
    
    #构建当前结点左分支
    if len(x0)==0 or score(root.model, x0, y0)>= precision:
        #左分支训练集为空或当前结点的线性模型对左分支的训练样本精度达到了阈值要求(precision),将左分支构建为叶子节点
        root.left = TreeNode(C=-1)
    else:
        #左分支结点精度不够要求,还需进行划分
        root.left = TreeNode()
        treeGenerate(root.left, x0, y0, precision)
    
    #构建当前结点右分支
    if len(x1)==0 or score(root.model, x1, y1) >= precision:
        root.right = TreeNode(C=1)
    else:
        root.right = TreeNode()
        treeGenerate(root.right, x1, y1, precision)

def predict(root, xs):
    #使用以root为根结点的决策树预测样本s
    if root.C is not None:
        #root为叶子结点
        return root.C
    else:
        if root.model.predict(np.expand_dims(xs, axis=0)) <= 0:
            return predict(root.left, xs)
        else:
            return predict(root.right, xs)

def evaluate(root, x, y):
    #计算以root为根结点的决策树在数据集x上的精度
    right = 0
    for i in range(x.shape[0]):
        if predict(root, x[i]) == y[i]:
            right += 1
    return right / x.shape[0]

if __name__ == '__main__':
    # 加载西瓜数据集3.0 alpha
    X = np.array([[0.697,0.46],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],[0.719,0.103]]) 
    Y = np.array([1]*8+[-1]*9)
    #参数random_state是指随机生成器,测试集占全部数据的33%
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
    
    #构建决策树
    root = TreeNode()
    #此处的阈值不能设的太大,由于数据本身就有一定客观存在的误差,无法做到100%精度,阈值设的太大容易爆栈
    treeGenerate(root, X, Y, 0.85)
    
    #计算训练好的决策树在测试集上的精度
    scoreTest = evaluate(root, X, Y)

    print('测试集精度为:', round(scoreTest, 4))
    
    # 展示决策树的分类向量
    x_val = np.linspace(0, 1, 100)
    print(params)
    params = np.array(params)
    for i in range(params.shape[0]):
        y_val = -(params[i][0]*x_val+params[i][2])/params[i][1]
        plt.plot(x_val, y_val)
    
    # 绘图，展示决策树的分类效果
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.show()
    
    

