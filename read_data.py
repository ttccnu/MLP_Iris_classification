
with open("iris.txt") as f:
    dataset = []
    iris = f.readlines()
    for i in iris:
        dataset.append(i.strip().split(' '))
    f.close()


dataset = dataset[1::]
data = []
label = []
for i in dataset:
    temp = []
    for m in i[1:5]:
        temp.append(float(m))
    data.append(temp)
    label.append(i[5])

category = ['"setosa"', '"versicolor"', '"virginica"']

label_onehot = []

for i in label:
    if i == category[0]:
        label_onehot.append([1, 0, 0])
    elif i == category[1]:
        label_onehot.append([0, 1, 0])
    elif i == category[2]:
        label_onehot.append([0, 0, 1])


def load_data():
    return data[0:20], label_onehot[0:20], data[20::], label_onehot[20::]
