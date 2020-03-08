# tian@mails.ccnu.edu.cn  2020/03/06

from model import MLP
from read_data import load_data
import random
import matplotlib.pyplot as plt


test_data, test_label, train_data, train_label = load_data()

test_set = []
for m in zip(test_data, test_label):
    test_set.append([m[0], m[1]])

nn = MLP(4, 2, 3, h_w=[random.random() for _ in range(8)], h_b=0.3,
         o_w=[random.random() for _ in range(6)], o_b=0.6, learn_rate=0.2)

record = []

print("正在训练中....请稍后")
for epoch in range(1000):
    for i in zip(train_data, train_label):
        nn.train(i[0], i[1])
    if epoch % 10 == 0:
        error = round(nn.total_error(test_set), 9)
        # print("训练轮数: ", epoch, "\n测试集误差: ", error)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        record.append(error)

print("训练结束，请输入鸢尾花的属性...")

plt.plot(record)
# plt.show()


def load_trained_model():
    return nn

