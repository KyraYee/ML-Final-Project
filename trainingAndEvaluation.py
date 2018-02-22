
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def performance(y_true, y_pred, metric="specificity") :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
   
    y_label=y_pred
    
    if metric=="accuracy":
        return metrics.accuracy_score(y_true,y_label)
    elif metric=="f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric=="auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric=="precision":
        return metrics.precision_score(y_true, y_label)
    elif metric=="sensitivity":
        return metrics.recall_score(y_true, y_label)
    elif metric=="specificity":
        cm=metrics.confusion_matrix(y_true,y_label)
        tn, fp, fn, tp = cm.ravel()
        return tn/float(tn+fp)
    
    cm=metrics.confusion_matrix(y_true,y_label)
    tn, fp, fn, tp = cm.ravel()
    return tn/float(tn+fp)

def test_performance(clf,x_test,y_test):
    """
    Calculates the performance metrics on the test set given a pretrained classifier
    
    Parameters
    --------------------
        clf    -- a pretrained classifier
        x_test -- numpy array of shape (n,d), the test data
        y_test -- numpy array of shape (n,), the test labels        
    
    Returns
    --------------------
        prec -- float, the precision
        rec -- float, recall
        acc -- float, accuracy
        spec --float, specificity
        f1 -- float, f1 score
        cm --np array, confusion matrix
    """

    y_pred=clf.predict(x_test)
    acc=metrics.accuracy_score(y_test,y_pred)
    cm=metrics.confusion_matrix(y_test,y_pred)
    prec=metrics.precision_score(y_test, y_pred)
    rec=metrics.recall_score(y_test, y_pred)
    f1=metrics.f1_score(y_test,y_pred)
    spec=performance(y_test, y_pred, metric="specificity") 


    return  prec, rec, acc, spec, f1,  cm

def train_and_test(clf, X_tr, y_train, skf, X_te, y_test, standardized=False):
    """
    Trains a model, calculates cross validation metrics, and test metrics
    
    Parameters
    --------------------
        clf    -- a classifier
        x_test -- numpy array of shape (n,d), the test data
        y_test -- numpy array of shape (n,), the test labels        
    
    Returns
    --------------------
        prec -- float, the precision
        rec -- float, recall
        acc -- float, accuracy
        spec --float, specificity
        f1 -- float, f1 score
        cm --np array, confusion matrix
    """
    if not standardized:
        print("Doing feature standardization")
        scaler=StandardScaler()
        X_train = scaler.fit_transform(X_tr)
        X_test=scaler.transform(X_te)
    else:
        X_train=X_tr
        X_test=X_te



    cv_prec, cv_rec, cv_acc, cv_f1, cv_spec=cv_performance(clf, X_train, y_train, skf)
    print("CV prec:", cv_prec)
    print("CV rec:", cv_rec)
    print("CV acc:", cv_acc)
    print("CV f1:", cv_f1)
    print("CV spec:", cv_spec)

    prec, rec, acc, spec, f1, cm =test_performance(clf, X_test, y_test)
    print("confusion matrix test set:")
    print(cm)
    print("test prec:", prec)
    print("test rec:",rec)
    print("test acc:", acc)
    print("Test spec:", spec)
    print("test f1:",f1)


def cv_performance(clf, X, y, kf) :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        
    Returns
    --------------------
        Returns the following metrics, all averaged across folds:

        precision
        recall
        accuracy
        f1
        specificity
    """
    
    precision = []
    recall=[]
    accuracy=[]
    f1_lst=[]
    specificity=[]

    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc=metrics.accuracy_score(y_test,y_pred)
        prec=metrics.precision_score(y_test, y_pred)
        rec=metrics.recall_score(y_test, y_pred)
        f1=metrics.f1_score(y_test, y_pred)
        spec=performance(y_test, y_pred, metric="specificity") 

       
        specificity.append(spec)
        precision.append(prec)
        recall.append(rec)
        accuracy.append(acc)
        f1_lst.append(f1)


        
    return np.array(precision).mean(), np.array(recall).mean(), np.array(accuracy).mean(), np.array(f1_lst).mean(), np.array(specificity).mean()

def evaluate(clf,  X_test, y_test):
    #take in already fitted model
    #evaluate on test set
    #return precision, recall, and confusion matrix
    
    y_pred=clf.predict(X_test)
    recall=recall_score(y_test, y_pred, average='weighted')
    precision=precision_score(y_test, y_pred, average='weighted')
    cm=confusion_matrix(y_test,y_pred)
    return recall, precision, cm




def analyzeMisclassified(y_pred, y_test, X_test, filename):

    fp_indices=[]
    fn_indices=[]
    correct_indices=[]
    incorrect_indices=[]
    for index in range(len(y_pred)):
        if y_pred[index]==y_test[index]:
            correct_indices.append(index)
        elif y_pred[index]==1 and y_test[index]==0:
            fp_indices.append(index)
            incorrect_indices.append(index)
        elif y_pred[index]==0 and y_test[index]==1:
            fn_indices.append(index)
            incorrect_indices.append(index)

    fp=X_test[fp_indices]
    fn=X_test[fn_indices]
    incorrect=X_test[incorrect_indices]
    correct=X_test[correct_indices]

    np.savez_compressed(filename, fp=fp, fn=fn, incorrect=incorrect, correct=correct)
   
