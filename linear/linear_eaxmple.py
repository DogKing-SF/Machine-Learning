import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model #获取线性模型

def read_data(path):
    """
    pandas读取数据
    path：string：数据路径
    return：dataframe，数据建模
    """
    return pd.read_csv(path)

def train_model(x,y):
    """
    利用训练数据，估计模型参数
    x：DataFrame，特征
    y：DataFrame，标签
    model：训练好的线性模型
    """
    model=linear_model.LinearRegression()
    model.fit(x,y)
    return model

def predict(model,x,y):
    #均方差(the mean squared error)
    mse = np.mean((model.predict(x)-y)**2)
    #决定系数
    score = model.score(x,y)
    return mse,score

def visuliaze_model(model,x,y):
    """
    模型可视化
    """
    fig = plt.figure(figsize=(6,6),dpi=80)
    ax = fig.add_subplot(111)
    ax.scatter(x,y,color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x,model.predict(x),color='red',label=u'$y=%.3fx+%.3f$'%(model.coef_.item(), model.intercept_.item()))
    plt.legend(shadow=True)
    plt.show()

def run_module(data):
    """
    data:DataFrame，建模数据
    """
    features=['x']
    labels=['y']
    model=train_model(data[features],data[labels])
    mse,score=predict(model,data[features],data[labels])
    print("MSE is %f"% mse)
    print("Score is %f"% score)
    visuliaze_model(model,data[features],data[labels])


if __name__ == "__main__":
    home_path=os.path.dirname(os.path.abspath(__file__))
    if os.name=="nt":
        data_path="%s\\simple_example.csv"%home_path
    else:
        data_path="%s/simple_example.csv"%home_path
    data = read_data(data_path)
    run_module(data)