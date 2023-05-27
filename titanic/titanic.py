import numpy as np
import pandas as pd

#学習データ、テストデータの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#学習データを特徴量と目的変数に分ける
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

#テストデータは特徴量のみなのでそのままでいい
test_x = test.copy()

from sklearn.preprocessing import LabelEncoder

#変数PassengerIdを除外する
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)


#変数Name, Ticket, Cabinを除外する
train_x  = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x  = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#それぞれのカテゴリ変数にlabel encordingを適用する
for c in ['Sex', 'Embarked']:
    #学習データに基づいてどう変換するかを定める
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))
    
    #学習データ、テストデータを変換する
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))


from xgboost import XGBClassifier

#モデルの作成および学習データを与えての学習
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

#テストデータの予測値を確率で出力
pred = model.predict_proba(test_x)[:, 1]

#テストデータの予測値を二値に変換する
pred_label = np.where(pred > 0.5, 1, 0)

    
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold 

#各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

#クロスバリデーションを行う
#学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    #学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    #モデルの学習を行う
    model = XGBClassifier(n_estimator=20, random_state=71)
    model.fit(tr_x, tr_y)
    
    #バリデーションデータの予測値を確率で出力する
    va_pred = model.predict_proba(va_x)[:, 1]
    
    #バリデーションデータのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)
    
    #そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

#各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss:{logloss:.4f}, accuracy:{accuracy:.4f}')

import itertools

#チューニング候補とするパラメータを準備する
param_space = {
    'max_depth':[3, 5, 7],
    'min_child_weight':[1.0, 2.0, 4.0]
}

#探索するハイパーパラメータの組み合わせ
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

#各パラメータの組み合わせ、それに対するスコアを保存するリスト
params = []
scores = []

#各パラメータの組み合わせごとに、クロスバリデーションで評価を行う
for max_depth, min_child_weight in param_combinations:
    
    score_folds = []
    #クロスバリデーションを行う
    #学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        #学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
        #モデルの学習を行う
        model = XGBClassifier(n_estimator=20, random_state=71)
        model.fit(tr_x, tr_y)
    
        #バリデーションデータのスコアを計算し、保存する
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)
    
    #各foldのスコアを平均する
    score_mean = np.mean(score_folds)
    
    #パラメータの組み合わせ、それに対するスコアを保存する
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

#最もスコアが良いものをベストなパラメータとする
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')

#max_depth=7, min_child_weight=2.0のスコアが一番良かった

from sklearn.linear_model import LogisticRegression

#xgboostモデル
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model.xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

#ロジスティック回帰モデル
#xgboostモデルとは異なる特徴量を入れる必要があるので、別途にtrain_x2, test_x2を作成した

#提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submmission_first.csv', index=False)