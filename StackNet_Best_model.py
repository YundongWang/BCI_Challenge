"""
@author: Yundong Wang
"""

from pystacknet.pystacknet import StackNetClassifier
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

# load data
X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')
y_train = pd.read_csv('./data/TrainLabels.csv')['Prediction'].values
y_test = np.reshape(pd.read_csv('./data/true_labels.csv', header=None).values, 3400)

C_list = [100,10, 1, 0.1]

models =[
    #1ST layer #
    [GridSearchCV(LDA(solver = 'lsqr'),  {'shrinkage':(0.01, 0.1, 1.)}, cv=3, n_jobs=-1) ,
     GridSearchCV(SVC(class_weight='balanced', probability = True),  {'kernel': ['rbf'],'gamma': [10**(-2), 10**(-3)],'C': C_list}, cv=3, n_jobs=-1),
     GridSearchCV(LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter = 100000), {'C': C_list}, cv =3, n_jobs=-1),
     LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.01, n_estimators=1000, class_weight='balanced', subsample_for_bin=1000, objective="xentropy", min_split_gain=0.0, min_child_weight=0.01, min_child_samples=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.5, reg_alpha=0.0, reg_lambda=0.0, random_state=1, n_jobs=1),
     GaussianProcessClassifier(),
     XGBClassifier(max_depth=5,learning_rate=0.3, reg_lambda=0.1, n_estimators=300, objective="binary:logistic", n_jobs=1, booster="gblinear", random_state=1, colsample_bytree=0.4 ),
     XGBClassifier(max_depth=20,learning_rate=0.1, n_estimators=300, objective="binary:logistic", n_jobs=1, booster="gbtree", random_state=1, colsample_bytree=0.4 ),
     XGBClassifier(max_depth=100,learning_rate=0.1, n_estimators=300, objective="rank:pairwise", n_jobs=1, booster="gbtree", random_state=1, colsample_bytree=0.4 ),
    ],

     #2ND layer #
    [
     RandomForestClassifier(max_depth = 50, n_estimators=50),
     AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=5), n_estimators = 200),
     ExtraTreesClassifier(max_depth=5, n_estimators = 50),
    ],

    #3RD layer #
    [
    RandomForestClassifier(max_depth = 5, class_weight = 'balanced', n_estimators=100),
    SVC(kernel='linear', class_weight='balanced', probability = True),
    MLPClassifier(hidden_layer_sizes=(32,16), activation="relu", solver="sgd",alpha=0.01,
                 batch_size=30, learning_rate="adaptive",learning_rate_init=0.001, power_t=0.5,
                 max_iter=100, shuffle=True, random_state=1, tol=0.0001,nesterovs_momentum = True, momentum=0.9,validation_fraction=0.1, early_stopping = True,
                 beta_1=0.1, beta_2=0.1, epsilon=0.1)
    ]
]

# leave 4 subject out
kf = KFold(4)
generator = kf.split(X_train, y_train)

# build StackNet
model = StackNetClassifier(models, metric="auc", folds=generator, restacking=False,
                             use_retraining=True, use_proba=True, random_state=42,
                             n_jobs=-1, verbose=1)
# evaluate model
model.fit(X_train,y_train)
y_probs = model.predict_proba(X_test)[:,1]

# save score
csv = pd.read_csv('./data/benchmark.csv')
csv['Prediction'] = y_probs
csv.to_csv('submission_StackNet.csv', index=False)

print('--------------------Submission file has been generated.--------------------------')
