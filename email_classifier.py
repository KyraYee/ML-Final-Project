import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score, precision_score, precision_recall_curve, roc_curve, confusion_matrix,f1_score
import pickle
from random import randint
from imblearn.under_sampling import RandomUnderSampler
from sklearn import model_selection
import argparse
import os
from mbox_reader import *

from nltk.stem.snowball import SnowballStemmer


def performance(y_true, y_pred, metric="accuracy") :
    """
    #from ps 6
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
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
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
    
    return 0

def extract_feature_vectors(data, word_list) :
    """
   
    
    Parameters
    --------------------
        data            --list of dicts
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of data points
                            d is the number of unique words in the text file
    """
    
    num_lines = len(data)
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
   
       
    #sorted_word_list=sorted(word_list.items(), key=lambda it: it[1]) # the dictionary sort by its values
    vector_list=[]
    for email in data:
        email_vector=[]
        stemmer = SnowballStemmer("english")
        email_tokens=email['tokens']
        try:
            email_words= [stemmer.stem(word) for word in email_tokens]
            

            for word in word_list: # we are ignoring words not in the dictionary
            
                if word in email_words:
                    email_vector.append(1)
                   
                else:
                    email_vector.append(0)
            vector_list.append(email_vector)
        except UnicodeDecodeError:
            print("unicode error in extract feature vectors ")
   

    feature_matrix=np.array(vector_list)
    
    
    return feature_matrix #, sorted_word_list

def test_performance(clf,x_test,y_test):
    y_pred=clf.predict(x_test)
    acc=metrics.accuracy_score(y_test,y_pred)
    cm=metrics.confusion_matrix(y_test,y_pred)
    prec=metrics.precision_score(y_test, y_pred)
    rec=metrics.recall_score(y_test, y_pred)
    f1=metrics.f1_score(y_test,y_pred)
    spec=performance(y_test, y_pred, metric="specificity") 


    return  prec, rec, acc, spec,f1,  cm

def train_and_test(clf, X_tr, y_train, skf, X_te, y_test, parameters=None):
    
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_tr)
    X_test=scaler.transform(X_te)


    cv_prec, cv_rec, cv_acc=cv_performance(clf, X_train, y_train, skf)
    print("CV prec", cv_prec)
    print("CV rec", cv_rec)
    print("CV acc", cv_acc)
    print("CV f1", 2*cv_prec*cv_rec/float(cv_prec+cv_rec))

    prec, rec, acc, spec, f1, cm =test_performance(clf, X_test, y_test)
    print("confusion matrix test set")
    print(cm)
    print("test prec", prec)
    print("test rec",rec)
    print("test acc", acc)
    print("Test spec", spec)
    print("test f1",f1)

def cv_performance(clf, X, y, kf) :
    """
    from ps6
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
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    precision = []
    recall=[]
    accuracy=[]

    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
#        score = performance(y_test, y_pred, metric)
        acc=metrics.accuracy_score(y_test,y_pred)
        #cm=metrics.confusion_matrix(y_true,y_pred)
        prec=metrics.precision_score(y_test, y_pred)
        rec=metrics.recall_score(y_test, y_pred)

        # if not np.isnan(score) :
        #     scores.append(score)

        precision.append(prec)
        recall.append(rec)
        accuracy.append(acc)
        
    
        
    return np.array(precision).mean(), np.array(recall).mean(), np.array(accuracy).mean()

def evaluate(clf,  X_test, y_test):
    #take in already fitted model
    #evaluate on test set
    #return precision, recall, and confusion matrix
    
    y_pred=clf.predict(X_test)
    recall=recall_score(y_test, y_pred, average='weighted')
    precision=precision_score(y_test, y_pred, average='weighted')
    cm=confusion_matrix(y_test,y_pred)
    return recall, precision, cm




# def trainAndEvaluate(X_train_unbalanced, y_train,X_test, y_test, balance_dict, parameters_dict):

#     rus = RandomUnderSampler(random_state=42, ratio=balance_dict)
#     X_res, y_train = rus.fit_sample(X_train_unbalanced, y_train )
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_res)
#     print("xtrain shape")
#     print(X_train.shape)
#     print("y train counts")
#     print(np.unique(y_train,return_counts=True))

#     RF = RandomForestClassifier(n_jobs=-1,**parameters_dict)
#     RF.fit(X_train,y_train)
#     X_test=scaler.transform(X_test)
#     y_pred=RF.predict(X_test)
#     y_pred_train=RF.predict(X_train)   
#     rec=recall_score(y_train, y_pred_train, average='weighted')
#     prec=precision_score(y_train, y_pred_train, average='weighted')
#     cm=confusion_matrix(y_train,y_pred_train)


#     test_rec,test_prec,test_cm=evaluate(RF, X_test, y_test)

#     return RF, rec, prec ,cm, test_rec, test_prec, test_cm


def selectParamLogReg(X,y, skf):
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

def findBestHyperParameterRF(X, y, parameter, parameter_values, known_parameters_dict=None):
    #test each value of parameter on one fold   
    plt.figure()
    splits=len(parameter_values)
    index=0
    f1_lst=[]
    

    kf = model_selection.KFold(n_splits=splits, shuffle=True)

    
    for train_index, val_index in kf.split(X):
        scaler = StandardScaler()
        kwargs = {parameter:parameter_values[index]}
        kwargs.update(known_parameters_dict)
        RF = RandomForestClassifier(n_jobs=-1,  **kwargs)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
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


def findNonBiasedFeatures(feature_importances, threshold):
    #give indices of all features whose importance is under the threshold
    unbiased_features=[]
    biased_features=[]
    for index in range(len(feature_importances)):
        if feature_importances[index]>threshold:
            biased_features.append(index)
        else:
            unbiased_features.append(index)
    return unbiased_features, biased_features






def RandomForestHyperTuning(X, y ):

    hyperparameters={}
    max_depth=range(20,120,5)
    findBestHyperParameterRF(X, y, "max_depth", max_depth, {})

    choice = input("input max depth parameter ")
    hyperparameters["max_depth"]=choice

    n_estimators = np.arange(5, 50, 4)
    findBestHyperParameterRF(X, y, "n_estimators", n_estimators, {})

    choice = input("input n_estimators parameter ")
    hyperparameters["n_estimators"]=choice

    end_Range=min(X.shape[1],100)

    features_range = range(1, end_Range, 5)
    findBestHyperParameterRF(X, y, "max_features", features_range, {})

    choice = input("input max_features parameter ")
    hyperparameters["max_features"]=choice

    min_samples_split = range(20, 100, 5)
    findBestHyperParameterRF(X, y, "min_samples_split", min_samples_split, {})

    choice = input("input min_samples_split parameter ")
    hyperparameters["min_samples_split"]=choice

    min_samples_leaf = range(1, 500, 30)
    findBestHyperParameterRF(X, y, "min_samples_leaf", min_samples_leaf, {})       
    choice = input("input min_samples_leaf parameter ")
    hyperparameters["min_samples_leaf"]=choice


    return hyperparameters




def generateBOWData():
     #1 for chat, 0 for dorm
    chat_data=loadData('north-chat.mbox') 
    dorm_data=loadData('north-dorm.mbox')
    print("loaded data")

    inits1,replies1=separateReplies(chat_data)
    inits2, replies2=separateReplies(dorm_data)

    inits1=[x for x in inits1 if x['body']!=None]
    inits2=[x for x in inits2 if x['body']!=None]
    print("len chat inits", len(inits1))
    print("len dorm inits", len(inits2))
    print("separated replies")

    y1=np.full(len(inits1),1)
    y2=np.full(len(inits2),0)
    y=np.concatenate((y1,y2), axis=0)

    X_unprocessed= inits1+inits2

    assert len(X_unprocessed)==y.shape[0]
#wo train test split

    X, dictionary= clean_text(X_unprocessed)
    print("len dictionary", len(dictionary))
    X=extract_feature_vectors(X, dictionary)
    print("training data shape", X.shape)

    np.savez_compressed("init_only_data" , X=X, y=y)
#w train test split
    # X_train, X_test, y_train, y_test = train_test_split(X_unprocessed, y, test_size=0.15, random_state=47, shuffle=True)
   

   # #process training
   #  X_train, dictionary= clean_text(X_train)
   #  print("len dictionary", len(dictionary))
    
   #  X_train=extract_feature_vectors(X_train, dictionary)
   #  print("training data shape", X_train.shape)

   #  []

   #  #process testing
   #  X_test=clean_text_no_dict(X_test)
   #  X_test=extract_feature_vectors(X_test, dictionary)
   #  print("test data shape", X_test.shape)

    

    
   #  np.savez_compressed("init_only_data_train" , X=X_train, y=y_train)
   #  np.savez_compressed("init_only_data_test" , X=X_test, y=y_test)




def pltTimeData():
    #1 for chat, 0 for dorm
    chat_data = loadData('north-chat.mbox') 
    dorm_data = loadData('north-dorm.mbox')

    inits1, replies1 = separateReplies(chat_data)
    inits2, replies2 = separateReplies(dorm_data)
   
    doW1 = []   
    dt_objs1 = []
    for email in inits1:
        r = re.compile('.{2}:.{2}:.{2}')
        time_sent = int(email['Date'][-14:-12])
        doW = email['Date'][:3]
        doW1.append(doWtoNum(doW))
        # if r.match(time_sent):
            # dt_obj = dt.datetime.strptime(time_sent, '%H:%M:%S')
        if time_sent < 24:
            dt_objs1.append(time_sent)

    doW2 = []
    dt_objs2 = []
    for email in inits2:
        r = re.compile('.{2}:.{2}:.{2}')
        time_sent = int(email['Date'][-14:-12])
        doW = email['Date'][:3]
        doW2.append(doWtoNum(doW))
        # if r.match(time_sent):
            # dt_obj = dt.datetime.strptime(time_sent, '%H:%M:%S')
        if time_sent < 24:
            dt_objs2.append(time_sent)


    # plt.figure()
    # plt.hist(dt_objs1, alpha=0.5, label='chat')
    # plt.hist(dt_objs2, alpha=0.5, label='dorm')
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.hist(doW1, alpha=0.5, label='chat')
    plt.hist(doW2, alpha=0.5, label='dorm')
    plt.legend()
    plt.show()


    date = 'Fri, 6 May 2016 18:49:54 -0700'





def generateBOWTimeData( ):
     #1 for chat, 0 for dorm
    chat_data=loadData('north-chat.mbox') 
    dorm_data=loadData('north-dorm.mbox')
    print("loaded data")

    inits1,replies1=separateReplies(chat_data)
    inits2, replies2=separateReplies(dorm_data)

    inits1=[x for x in inits1 if x['body']!=None]
    inits2=[x for x in inits2 if x['body']!=None]
    print("len chat inits", len(inits1))
    print("len dorm inits", len(inits2))
    print("separated replies")

    y1=np.full(len(inits1),1)
    y2=np.full(len(inits2),0)
    y=np.concatenate((y1,y2), axis=0)

    X_unprocessed= inits1+inits2

    assert len(X_unprocessed)==y.shape[0]

    #get BOW
    raw_X, dictionary= clean_text(X_unprocessed)
    print("len dictionary", len(dictionary))
    X=extract_feature_vectors(raw_X, dictionary)
    print(" data shape", X.shape)

    #get day of week and time
    time_data=[]
    for email in raw_X:
        time_sent = int(email['Date'][-14:-12])

        doW = email['Date'][:3]
        doW=doWtoNum(doW)
        time_data.append([doW, time_sent])

    time_data=np.array(time_data)
    assert time_data.shape[0]==X.shape[0]

    #combine
    Xfinal=[]
    for index in range(X.shape[0]):
        Xfinal.append(np.concatenate((time_data[index],X[index]), axis=0))

    Xfinal=np.array(Xfinal)
    print("data with time shape", Xfinal.shape)
    
    np.savez_compressed("init_only_data_w_time" , X=Xfinal, y=y, raw_X=raw_X, dictionary=dictionary )







def main(model):
    a=np.load("init_only_data_w_time.npz")
    
    X=a['X']
    
    y=a['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=47, shuffle=True)
    #1 for chat, 0 for dorm
    # train=np.load("init_only_data_train.npz")
    
    # X_train=train['X']
    
    # y_train=train['y']

    # test=np.load("init_only_data_test.npz")
    # X_test=test['X']

    # y_test=test['y']

    print("extracted feature vectors")
   
    skf= StratifiedKFold(n_splits=10, shuffle=True)

    if model=="log reg":
        c=selectParamLogReg(X_train,y_train, skf)
        LR=LogisticRegression(C=c)
       
       
        print("log reg")
        print(train_and_test(LR, X_train, y_train, skf, X_test, y_test))

    elif model=="RF":
        RFparameters=RandomForestHyperTuning(X_train, y_train) #need to use these to train
        RFtuned= RandomForestClassifier(**RFparameters)
        print("random forest")
        print(train_and_test(RFtuned, X_train, y_train, skf, X_test, y_test))
   
   




parser = argparse.ArgumentParser(description='Run classification')
parser.add_argument('--model', dest='model', type=str, default='log reg', help='type of classifier to use')
#parser.add_argument('--dataWithFips', dest='dataWithFips', type=str, default='data/census_data_filtered_disability_with_fips.npz', help='The data set in dictionary fo')


if __name__ == "__main__": 
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)