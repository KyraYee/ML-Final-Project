import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import make_classification
import pickle
from random import randint
from imblearn.under_sampling import RandomUnderSampler
import argparse
import os
from mbox_reader import *
from trainingAndEvaluation import *
from parameterSelection import *
from data_preprocessing import *
from sklearn.decomposition import PCA




def main(model, pca, filename):
    a=np.load(filename)
    
    X=a['X']
    
    y=a['y']


    print(np.unique(y, return_counts=True))
    print("x shape", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=47, shuffle=True)

    print("extracted feature vectors")
   
    skf= StratifiedKFold(n_splits=10, shuffle=True)

    standardized=False

    if model=="adaboost":

        n=selectParamAdaboost(X, y, standardized=standardized)
        #n=65
        clf=AdaBoostClassifier(n_estimators=n)
        print("Adaboost")
        print(train_and_test(clf, X_train, y_train, skf, X_test, y_test))
        

        with open("adaboost.pkl","wb") as f:
            pickle.dump(clf, f)

        np.savez_compressed("test_train_data", X_train=X_train, y_train= y_train, X_test=X_test, y_test=y_test)
       
    if model=="log reg":
        
        c=selectParamLogReg(X_train,y_train, skf)
        #c=1
        LR=LogisticRegression(C=c)
       
       
        print("log reg")
        print(train_and_test(LR, X_train, y_train, skf, X_test, y_test, standardized=standardized))

    elif model=="RF":
        print("rf hyperparameter tuning")
        RFparameters=RandomForestHyperTuning(X_train, y_train, standardized=False) #need to use these to train
        #RFparameters={"max_depth":100, "n_estimators":45, "max_features":31, "min_samples_split":80, "min_samples_leaf":5}
       
        RFtuned= RandomForestClassifier(**RFparameters)
        print("random forest")
        print(train_and_test(RFtuned, X_train, y_train, skf, X_test, y_test))
    

    elif model=="SVM":

        SVMparameters=tuneSVM(X_train, y_train, standardized=True)
        # c=0.01
        # gamma=0.1
        gamma=SVMparameters[1]
        c=SVMparameters[0]
        SVM = SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
        train_and_test(SVM, X_train, y_train, skf, X_test, y_test)


        with open("SVMrbf.pkl","wb") as f:
            pickle.dump(SVM, f)




parser = argparse.ArgumentParser(description='Run classification')
parser.add_argument('--model', dest='model', type=str, default='RF', help='type of classifier to use')
parser.add_argument('--filename', dest='filename', type=str, default="init_only_data_time_attach.npz", help='the preprocessed data file to use')




if __name__ == "__main__": 
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)