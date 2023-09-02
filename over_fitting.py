import numpy as np
import matplotlib.pyplot as plt


# y = w0*x^0+....wM*x^M　を、引数xの配列数分求める
def y(w, x, M):
    X = np.empty((M + 1, x.size))
    for i in range(M + 1):
        X[i, :] = x ** i
    return np.dot(w.T, X)


# ランダムシードを固定
np.random.seed(0)
# 多項式の最大べき乗数(x^0+...+x^M)
M = 9
# 訓練データ数
N = 10

# 正則化係数λ（参考書に倣った値）
lam = np.exp(-18)

# 訓練データの列ベクトル
x = np.linspace(0, 1, N).reshape(N, 1)
# 訓練データtの列ベクトル
t = np.sin(2*np.pi*x.T) + np.random.normal(0, 0.2, N)
t = t.reshape(N, 1)

# 行列Phiを作成
Phi = np.empty((N, M+1))
for i in range(M+1):
    Phi[:, i] = x.reshape(1, N) ** i

# 係数wの列ベクトルを解析的に求める
w = np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, t))

# 正則化項付きで求める
w2 = np.linalg.solve(np.dot(Phi.T, Phi) + lam*np.eye(M+1), np.dot(Phi.T, t))

# 求めた係数wを元に、新たな入力x2に対する予測値yを求める
x2 = np.linspace(0, 1, 100)
y1 = y(w, x2, M)
y2 = y(w2, x2, M)

# 結果の表示
plt.xlim(0.0, 1.0)
plt.ylim(-1.5, 1.5)
plt.scatter(x, t)
plt.plot(x2, y1.T)
plt.plot(x2, y2.T)
plt.show()