# coding: utf-8

import numpy as np # 配列
import time # 時間
from matplotlib import pyplot as plt # グラフ
import os # フォルダを作成可能（意外と便利）

# chainer
from chainer import Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def get_data(test_rate):
    data = np.genfromtxt("iris_e.csv", dtype=float, delimiter=",",skip_header=1,usecols = (0,1,2,3,4))

    x = data[:,0:4]
    t = data[:,4]

    x_test = x[:int(0.2*x.shape[0])]
    x_train = x[int(0.2*x.shape[0]):]
    t_test = t[:int(0.2*t.shape[0])]
    t_train = t[int(0.2*t.shape[0]):]

    return x_train, t_train, x_test, t_test

class IRIS(Chain):
    def __init__(self, h_units, act):
        super(IRIS, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(4, h_units[0])
            self.l2=L.Linear(h_units[0], h_units[1])
            self.l3=L.Linear(h_units[1], 3)

            if act == "relu":
                self.act = F.relu
            elif act == "sig":
                self.act = F.sigmoid

    def __call__(self, x, t):
        x = Variable(x.astype(np.float32).reshape(x.shape[0],4))
        t = Variable(t.astype("i"))
        y = self.forward(x)
        return F.softmax_cross_entropy(y, t), F.accuracy(y,t)

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        h = self.l3(h)
        return h


# 学習（学習パラメータを設定しやすくするために関数化）
def training(test_rate, bs, n_epoch, h_units, act):

    # データセットの取得
    x_train, t_train, x_test, t_test = get_data(test_rate)
    N = x_train.shape[0]
    Nte = x_test.shape[0]

    # モデルセットアップ
    model = IRIS(h_units, act)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # loss/accuracy格納配列
    tr_loss = []
    te_loss = []
    tr_acc = []
    te_acc = []

    # ディレクトリを作成
    if os.path.exists("Results/{}/".format(act)) == False:
        os.makedirs("Results/{}/".format(act))

    # 時間を測定
    start_time = time.time()
    print("START")

    # 学習回数分のループ
    for epoch in range(1, n_epoch + 1):
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_acc = []
        for i in range(0, N, bs):
            x_batch = x_train[perm[i:i + bs]]
            t_batch = t_train[perm[i:i + bs]]

            model.cleargrads()
            loss, acc = model(x_batch,t_batch)
            loss.backward()
            optimizer.update()
            sum_loss += loss.data * bs
            sum_acc.append(acc.data)

        # 学習誤差/精度の平均を計算
        ave_loss = sum_loss / N
        tr_loss.append(ave_loss)
        tr_acc.append(sum(sum_acc)/len(sum_acc))

        # テスト誤差
        loss, acc = model(x_test,t_test)
        te_loss.append(loss.data)
        te_acc.append(acc.data)

        # 学習過程を出力
        if epoch % 100 == 1:
            print("Ep/MaxEp     tr_loss     te_loss")

        if epoch % 10 == 0:
            print("{:4}/{}  {:10.5}   {:10.5}".format(epoch, n_epoch, ave_loss, float(loss.data)))

            # 誤差をリアルタイムにグラフ表示
            plt.plot(tr_loss, label = "training")
            plt.plot(te_loss, label = "test")
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss (Cross Entropy)")
            plt.pause(0.1)  # このコードによりリアルタイムにグラフが表示されたように見える
            plt.clf()


    print("END")

    # 経過時間
    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    # 誤差のグラフ作成
    plt.figure(figsize=(4, 3))
    plt.plot(tr_loss, label = "training")
    plt.plot(te_loss, label = "test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (Cross Entropy)")
    plt.savefig("Results/{}/loss_history.png".format(act))
    plt.clf()
    plt.close()

    # 精度のグラフ作成
    plt.figure(figsize=(4, 3))
    plt.plot(tr_acc, label = "training")
    plt.plot(te_acc, label = "test")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("acc (Cross Entropy)")
    plt.savefig("Results/{}/acc_history.png".format(act))
    plt.clf()
    plt.close()

    # 学習済みモデルの保存
    serializers.save_npz("Results/{}/Model.model".format(act),model)


if __name__ == "__main__":

    # 設定
    test_rate = 0.2
    bs = 10             # バッチサイズ
    n_epoch = 10       # 学習回数
    h_units = [10, 10]  # ユニット数 [中間層１　中間層２]
    act = "sig"        # 活性化関数

    training(test_rate, bs, n_epoch, h_units, act)
