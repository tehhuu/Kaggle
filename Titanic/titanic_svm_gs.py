import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# スコア : 0.75119

# 訓練データとテストデータの列数の差を埋める。
def fill_missing_columns(df_a, df_b):
    columns_for_b = set(df_a.columns) - set(df_b.columns)
    for column in columns_for_b:
        df_b[column] = 0
    columns_for_a = set(df_b.columns) - set(df_a.columns)
    for column in columns_for_a:
        df_a[column] = 0

# 訓練データの読み込み
df = pd.read_csv('Dataset/train.csv')

# 'Sex'をOne-Hotエンコーディング
dm_sex = pd.get_dummies(df['Sex'])
df = pd.concat([df, dm_sex], axis=1)

# Embarkedを補完後、One-Hotエンコーディング
#df['Embarked'] = df['Embarked'].fillna('S')
#dm_emb = pd.get_dummies(df['Embarked'])
#df = pd.concat([df, dm_emb], axis=1)

# 敬称を抜き出して列を作成後、One-Hotエンコーディング
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]
df["Title"] = pd.Series(dataset_title)
df["Title"].unique()
dm_ttl = pd.get_dummies(df["Title"])
df = pd.concat([df, dm_ttl], axis=1)

# 敬称毎に平均年齢を計算し、補完
title_list = list(df["Title"].unique())
avg_list = [df[df["Title"] == i]['Age'].mean() for i in title_list]
avg_dict = {title_list[i] : df[df["Title"] == title_list[i]]['Age'].mean() for i in range(len(avg_list))}
df['Age'] = df['Age'].fillna(-1)
for i in range(len(df)):
    if df['Age'][i] == -1:
        df['Age'][i] = avg_dict[df['Title'][i]]

# 訓練データを作成
X = np.array(df.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','female', 'male', 'Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess']].values)
y = np.array(df.loc[:, ["Survived"]].values)
y = y.reshape((1,-1))[0]

# 標準化
ss = StandardScaler()
X_std = ss.fit_transform(X)

# グリッドサーチSVMを用意
param_grid = [{'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001]}]
kf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
svc_for_gs = SVC(random_state=0, kernel='rbf')
gs_svc = GridSearchCV(svc_for_gs, param_grid, cv=kf_5)

# 学習
gs_svc.fit(X, y)


# テストデータを読み込み
df_test = pd.read_csv('Dataset/test.csv')
df_test_index = df_test.loc[:, ['PassengerId']]

#　================================
# 補完処理
dm_sex = pd.get_dummies(df_test['Sex'])
df_test = pd.concat([df_test, dm_sex], axis=1)

#df_test['Embarked'] = df_test['Embarked'].fillna('S')

dm_emb = pd.get_dummies(df_test['Embarked'])
df_test = pd.concat([df_test, dm_emb], axis=1)

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_test["Name"]]
df_test["Title"] = pd.Series(dataset_title)
df_test["Title"].unique()
dm_ttl = pd.get_dummies(df_test["Title"])
df_test = pd.concat([df_test, dm_ttl], axis=1)
df_test.isnull().sum()

title_list = list(df_test["Title"].unique())
avg_list = [df_test[df_test["Title"] == i]['Age'].mean() for i in title_list]

avg_dict = {title_list[i] : df_test[df_test["Title"] == title_list[i]]['Age'].mean() for i in range(len(avg_list))}
df_test['Age'] = df_test['Age'].fillna(-1)
for i in range(len(df_test)):
    if df_test['Age'][i] == -1:
        df_test['Age'][i] = avg_dict[df_test['Title'][i]]

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
#　================================

# 訓練データとテストデータの列数を揃える
fill_missing_columns(df, df_test)

# テストデータの作成と標準化
X_test = np.array(df_test.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','female', 'male', 'Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess']].values)
ss_test = StandardScaler()
X_test_std = ss_test.fit_transform(X_test)

# 予測
y_test = gs_svc.predict(X_test_std)

# 書き込み
df_output = pd.concat([df_test_index, pd.DataFrame(y_test, columns=['Survived'])], axis=1)
df_output.to_csv('result.csv', index=False)
