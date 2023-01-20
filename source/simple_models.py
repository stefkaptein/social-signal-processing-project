import model.load_data as ld
import model.scoring_metrics as sm

from sklearn import metrics
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from sklearn.svm import SVC 
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import xgboost as xgb
xgb.set_config(verbosity=0)
import matplotlib
from sklearn.model_selection import train_test_split 
pd.set_option('display.max_columns', None)
import sys
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot
import importlib
importlib.reload(sm)
importlib.reload(ld)

datasets = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002""".split(" ")
results_merged_path = "./results_merged_fixedf0/"

X_train, y_train, X_test, y_test = ld.train_test_split(datasets,results_merged_path,0.3)

all_features = ['pause', 'speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']
features_selected = ['similarity']

def filter(data):
    data['similarity'] = data['similarity'][2:-2]
    data['similarity'] = pd.to_numeric(data['similarity'])

    data.fillna(0,inplace=True)

    data = data[features_selected]
    
    return data

X_train = filter(X_train)
X_test = filter(X_test)

def print_eval(y_pred,y_true):
    k = int(max(1,np.floor((len(y_true)+1)/(2*(sum(y_true)+1)))))
    print('k =',k)

    int_y_pred = (np.array(y_pred))
    int_y_true = (np.array(y_true))

    print('- windiff:',sm.get_windiff(int_y_true,int_y_pred,k))
    print('- pk:',sm.get_pk(int_y_true,int_y_pred,k))
    print('- kkappa:',sm.get_k_kappa(int_y_true,int_y_pred,k))


def DecTree(X_train, X_test, y_train, y_test):
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    # get importance
    importance = clf.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature:',features_selected[i],'->',v)
    # plot feature importance
    pyplot.bar(features_selected, importance)
    pyplot.title('Feature Importance')
    pyplot.show()

    # plot tree
    plt.figure(figsize=(7,4))  # set plot size (denoted in inches)
    tree.plot_tree(clf, max_depth=3, fontsize=10,feature_names=features_selected)
    plt.show()
    
    return y_predicted

DT_y_predicted = DecTree(X_train,X_test,y_train,y_test)
print(DT_y_predicted.shape, y_test.shape)
print_eval(DT_y_predicted,y_test)