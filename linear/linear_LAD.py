#没有完成
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model #获取线性模型
from statsmodels.regression.quantile_regression import QuantReg


def create_data():
    np.random.seed(42)
    a=list(range(0,100))
    e=np.round(np.random.randn(100),2)
    return pd.DataFrame({'x':a, 'y':a+e})

def run_model(data):
    features=data[['x']]
    labels=data[['y']]
    model=train_model(features,labels)
    visualise(model, features, labels)
    return model

def train_model(features,labels):
    model = linear_model.LinearRegression()
    model.fit(features,labels)
    return model

def predict(model,features,labels):
    # mean of squared error
    mse = np.mean((model.predict(features)-labels)**2)
    # score
    return mse,model.score(features,labels)

def visualise(model,features,labels):
    fig = plt.figure(figsize=(10,10),dpi=520)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(features,labels,color='red')
    ax.plot(features,model.predict(features),color='blue',label='$y=%.3fx+%.3f$'%(model.coef_.item(), model.intercept_.item()))
    plt.legend(shadow=True)
    plt.show()

if __name__ == '__main__':
    data = create_data()
    model = run_model(data)