from train import load_trained_model

nn = load_trained_model()

category = ['"setosa"', '"versicolor"', '"virginica"']

while True:
    flower = list(map(float, input().split()))
    pred = nn.forward(flower)
    pre_cate = pred.index(max(pred))
    print("属性为: ", flower, "的预测花系为:", category[pre_cate])
    print("是否继续？ Y/N")
    if input() == "N":
        break
    else:
        pass
    pass


