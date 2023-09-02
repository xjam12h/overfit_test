import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge

import pandas as pd





# ランダムシードを固定
np.random.seed(0)
# 多項式の最大べき乗数(x^0+...+x^M)
M = 9
# データ数
N = 100
# 訓練データ数
TRAIN_N=int(N*0.7)-1

# 正則化係数λ（参考書に倣った値）
# alpha=np.exp(-18)
alpha=0.001

# 訓練データの列ベクトル
x = np.linspace(0, 1, N).reshape(N, 1)

# データtの列ベクトル
t = np.sin(12*np.pi*x.T)+x.T*6 + np.random.normal(0, 0.5, N)
t = t.reshape(N, 1)
t_d=pd.DataFrame(data=t,columns=[0])
# print(t_d.head())

# 行列Phiを作成
Phi = np.empty((N, M))

for i in range(M):
    Phi[:, i] = x.reshape(1, N) ** i

Phi_d=pd.DataFrame(data=Phi,columns=[i for i in range(M)])
# print(Phi_d.head())

column_name=[i for i in range(M)]
print(Phi_d[column_name].loc[:TRAIN_N],t_d[[0]].loc[:TRAIN_N])

print(Phi[:TRAIN_N,:].shape)

linear=LinearRegression()
linear.fit(Phi[:TRAIN_N,:],t[:TRAIN_N,:])
# linear.fit(Phi_d[column_name].loc[:TRAIN_N],t_d[[0]].loc[:TRAIN_N])

lasso=Lasso(alpha=alpha)
# lasso.fit(Phi_d[column_name].loc[:TRAIN_N],t_d[[0]].loc[:TRAIN_N])
lasso.fit(Phi[:TRAIN_N,:],t[:TRAIN_N,:])

ridge=Ridge(alpha=alpha)
# ridge.fit(Phi_d[column_name].loc[:TRAIN_N],t_d[[0]].loc[:TRAIN_N])
ridge.fit(Phi[:TRAIN_N,:],t[:TRAIN_N,:])

predict_linear=linear.predict(Phi)
predict_lasso=lasso.predict(Phi)
predict_ridge=ridge.predict(Phi)
    


# 結果の表示

plt.xlim(0.0, 1.0)
plt.ylim(-2, 10)
plt.scatter(x, t,c="b",label="data",alpha=0.6)
plt.plot(x,predict_linear,c="k",label="予測(Linear)",alpha = 0.8,linewidth=4)
plt.plot(x,predict_lasso,c="g",label="予測(Lasso)",linewidth=4,alpha = 0.8)
plt.plot(x,predict_ridge,c="r",label="予測(Ridge)",linewidth=4,alpha = 0.8)
plt.legend()
plt.show()


