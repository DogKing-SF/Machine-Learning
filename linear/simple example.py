import numpy as np  #数组
import pandas as pd   #数据框
import matplotlib.pyplot as plt #图形库
import os #

#生成数据并返回一个数据框
def generate_data():
    """
    随机数据生成器
    """
    #规定种子
    np.random.seed(42)

    x=np.array(list(range(10,30)))
    error=np.round(np.random.randn(20),2)
    y=x+error
    return pd.DataFrame({"x":x,"y":y})

#数据可视化
def visualize_data(data):
    """
    数据可视化
    :param data:
    :return:
    """
    fig = plt.figure(figsize=(6,6),dpi=80)#设定图形框大小和分辨率
    ax=fig.add_subplot(111)#缩写，设置只画一张图，，表示画布分成一行一列的第一幅图，参数为(rows=1,cols=1,index=1)
    ax.set_xlabel("$x$")#设定标签名
    ax.set_ylabel("$y$")
    ax.set_xticks(range(10,31,5))#设定刻度
    ax.set_yticks(range(10,31,5))
    ax.scatter(data.x,data.y,color="blue",label="$y=x + \\epsilon$")
    plt.legend(shadow=True)# 展示上一行label内容
    plt.show()# 展示图形

#存储数据
if __name__ == "__main__":
    data = generate_data()
    home_path = os.path.dirname(os.path.abspath(__file__))
    # 粗出数据，win和linux不同
    if os.name == "nt":
        data.to_csv("%s\\simple_example.csv" % home_path, index=False)#win
    else:
        data.to_csv("%s/simple_example.csv" % home_path, index=False)#linux
    visualize_data(data)