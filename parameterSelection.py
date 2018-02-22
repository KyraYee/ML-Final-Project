import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score


def tuneSVM(X,y, standardized=False):
    """
    Determines the best hyperparameters for a SVM
    
    Parameters
    --------------------
        X            -- numpy array of shape (n,d), the training data
        y            -- numpy array of shape (n,), the training labels
        standardized -- bool,whether or not standadization has been performed        
    
    Returns
    --------------------
        (bestC, bestGamma) -- (float, float) ,the C and gamma values that gave the highest performance
         on the validation set after being trained ona single fold of the training data
    """

    bestPerformance = 0
    bestTuple = (0, 0)
    gammaRange = 10.0**np.arange(-3, 4)
    CRange = 10.0**np.arange(-3, 4)
    index=0
    params=[]

    
    print("tuning SVM")
    for gamma in gammaRange:
        for c in CRange:
            params.append((gamma,c))

    #try each set of parameters on one fold
    kf = model_selection.KFold(n_splits=len(params), shuffle=True)
   
    
    for train_index, val_index in kf.split(X):
        c=params[index][1]
        gamma=params[index][0]
        clf = SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
       
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
       
        if not standardized:
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_val=scaler.transform(X_val)

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_val)
        f1=f1_score(y_val, y_pred, average='weighted')

        print("C = " + str(c) + ", gamma= "+str(gamma) +", performance = " + str(f1))
        score=f1
        index+=1

        if performance > bestPerformance:
            bestPerformance = performance
            bestC = c
            bestGamma=gamma      

    return (bestC, bestGamma)




def selectParamLogReg(X,y, skf, standardized=False):
    """
    Determines the best hyperparameters for a logistic regression
    
    Parameters
    --------------------
        X             -- numpy array of shape (n,d), the training data
        y             -- numpy array of shape (n,), the training labels
        standardized  -- bool,whether or not standadization has been performed        
    
    Returns
    --------------------
        bestC -- float, the C value that gave the highest performance
         on the validation set after being trained ona single fold of the training data
    """

    CRange = 10.0**np.arange(-3,4)
    bestC = 0
    bestPerformance = 0
    performances = []
    kf = model_selection.KFold(n_splits=len(CRange), shuffle=True)

    index=0

    for train_index, val_index in kf.split(X):
        c=CRange[index]
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = LogisticRegression(C=c, solver="sag", max_iter=700, n_jobs=-1)
        if not standardized:
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_val=scaler.transform(X_val)
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_val)
        f1=f1_score(y_val, y_pred, average='weighted')

        print("C = " + str(c) + ", performance = " + str(f1))
        performances.append(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestC = c

        index+=1

    return bestC

def selectParamAdaboost(X, y, standardized=False):
    """
    Determines the best hyperparameter for a adaboost

    Parameters
    --------------------
        X             -- numpy array of shape (n,d), the training data
        y             -- numpy array of shape (n,), the training labels  
        standardized  -- bool,whether or not standadization has been performed      
    
    Returns
    --------------------
        bestN -- the number of decision stumps that gave the highest performance
         on the validation set after being trained ona single fold of the training data
    """
    print("hyperparameter tuning adaboost")
    NRange=range(40,90,3)
    bestN = 0
    bestPerformance = 0
    performances = []
    kf = model_selection.KFold(n_splits=len(NRange), shuffle=True)

    index=0

    for train_index, val_index in kf.split(X):
        n=NRange[index]
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = AdaBoostClassifier(n_estimators=n)
        if not standardized:
            print("standardizing")
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_val=scaler.transform(X_val)
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_val)
        f1=f1_score(y_val, y_pred, average='weighted')

        print("n = " + str(n) + ", performance = " + str(f1))
        performances.append(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestN=n

        index+=1



    return bestN

def findBestHyperParameterRF(X, y, parameter, parameter_values, known_parameters_dict=None, standardized=False):
    """
    Determines the best hyperparameter for a given parameter for a random forest

    Parameters
    --------------------
        X                     -- numpy array of shape (n,d), the training data
        y                     -- numpy array of shape (n,), the training labels
        parameter             -- string, the name of the parameter we are trying to optimize
        parameter_values      -- list, the values of said parameter we want to test
        known_parameters_dict -- dictionary where the keys are strings of parameter names and values are parameter values. Other set parameters we can add to the CV model 
        standardized          -- bool,whether or not standadization has been performed
    
    Returns
    --------------------
        bestN -- the number of decision stumps that gave the highest performance
         on the validation set after being trained ona single fold of the training data
    """
    #test each value of parameter on one fold   
    plt.figure()
    splits=len(parameter_values)
    index=0
    f1_lst=[]
    

    kf = model_selection.KFold(n_splits=splits, shuffle=True)

    
    for train_index, val_index in kf.split(X):
        kwargs = {parameter:parameter_values[index]}
        kwargs.update(known_parameters_dict)
        RF = RandomForestClassifier(n_jobs=-1,  **kwargs)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if not standardized:
            scaler = StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_val=scaler.transform(X_val)
        RF.fit(X_train, y_train)
        y_pred=RF.predict(X_val)
        f1_lst.append(f1_score(y_val, y_pred, average='weighted'))
        index=index+1
       
       

    plt.scatter(parameter_values,f1_lst)
   
    plt.xlabel(parameter)
    plt.ylabel("f1 score")
   
    plt.show()



def RandomForestHyperTuning(X, y , standardized=False):
    """
    Determines the best hyperparameters for a random forest

    Parameters
    --------------------
        X                     -- numpy array of shape (n,d), the training data
        y                     -- numpy array of shape (n,), the training labels 
        standardized          -- bool,whether or not standadization has been performed
    
    Returns
    --------------------
        hyperparameters -- dictionary where the keys are strings of parameter names and values are parameter values, optimized via CV 
    """

    hyperparameters={}
    max_depth=range(5,100,3)
    findBestHyperParameterRF(X, y, "max_depth", max_depth, {},standardized)

    choice = input("input max depth parameter ")
    hyperparameters["max_depth"]=choice

    n_estimators = np.arange(5, 100, 4)
    findBestHyperParameterRF(X, y, "n_estimators", n_estimators, {},standardized)

    choice = input("input n_estimators parameter ")
    hyperparameters["n_estimators"]=choice

    end_Range=min(X.shape[1],25000)

    features_range = range(1, end_Range, 5)
    findBestHyperParameterRF(X, y, "max_features", features_range, {},standardized)

    choice = input("input max_features parameter ")
    hyperparameters["max_features"]=choice

    min_samples_split = range(4, 800, 30)
    findBestHyperParameterRF(X, y, "min_samples_split", min_samples_split, {},standardized)

    choice = input("input min_samples_split parameter ")
    hyperparameters["min_samples_split"]=choice

    min_samples_leaf = range(1, 600, 30)
    findBestHyperParameterRF(X, y, "min_samples_leaf", min_samples_leaf, {},standardized)       
    choice = input("input min_samples_leaf parameter ")
    hyperparameters["min_samples_leaf"]=choice


    return hyperparameters
