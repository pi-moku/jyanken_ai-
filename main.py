import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

# dfはデータフレームの略
df = pd.read_csv('data.csv')
# 平均値を求めて欠損値と置き換える
colmean = df.mean()
df2 = df.fillna(colmean)
df2.isnull().any(axis=0)

# 特徴量
col = ['四つ前に出した手', '三つ前に出した手', '二つ前に出した手', 'ひとつ前に出した手']
x = df2[col]
t = df2['出した手']

#学習データと訓練データで分割
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.3 ,random_state=0)
model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
#訓練データで学習
model.fit(x_train, y_train)

#評価
data_test = [[2,2,3,3]]
print(model.predict(data_test))
