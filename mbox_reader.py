import mailbox
import pandas as pd
import numpy as np
import nltk
from email.Header import decode_header
import email
from base64 import b64decode
import sys
from email.Parser import Parser as EmailParser
from email.utils import parseaddr
from StringIO import StringIO
import mailbox
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import pickle
from nltk.stem.snowball import SnowballStemmer

#from https://stackoverflow.com/questions/7166922/extracting-the-body-of-an-email-from-mbox-file-decoding-it-to-plain-text-regard
def getcharsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handleerror(errmsg, emailmsg,cs):
    print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",getcharsets(emailmsg))
    print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])

def getbodyfromemail(msg):
    body = None
    attachment=None
    #Walk through the parts of the email to find the text body.    
    if msg.is_multipart():    
        for part in msg.walk():
            attachment = parse_attachment(part)

            # If part is multipart, walk through the subparts.            
            if part.is_multipart(): 

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True) 
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif msg.get_content_type() == 'text/plain':
        attachment = parse_attachment(msg)
        body = msg.get_payload(decode=True) 

   # No checking done to match the charset with the correct part. 
    for charset in getcharsets(msg):
        try:
            body = body.decode(charset)
        except UnicodeDecodeError:
            handleerror("UnicodeDecodeError: encountered.",msg,charset)
        except AttributeError:
             handleerror("AttributeError: encountered" ,msg,charset)
        except UnicodeEncodeError:
            #body=body.decode(encoding='utf-8')
            body=body.encode('ascii', 'ignore')

    
    return body  , attachment


def makeStopWordList():
    stop_words=stopwords.words('english')
    extra=[".", ",", "<", ">", "@", "+", "=", "subscribe", "unsubscribe", ""]
    stop_words=stop_words+extra
    return stop_words

def clean_text(data):
    dictionary=[]
    stop_words=makeStopWordList()
    #takes in a list of dicts
    for entry in data:
        body=entry['body']
        tokens = nltk.word_tokenize(body)
        stemmer = SnowballStemmer("english")
        tokens=[word for word in tokens if word not in stop_words and len(word)<30]  #tokens are unicode
        for word in tokens:
            #word is of type unicode
            try:
                clean_word=stemmer.stem(word.encode('ascii', 'ignore'))
                if clean_word not in dictionary and "@" not in clean_word :

                    dictionary.append(clean_word)
            except UnicodeDecodeError:
                print("unicode error in text cleaning")
                print(word)
        entry['tokens']=tokens
    return data , dictionary


#'north-chat.mbox'
def loadData(mbox_file):
    data=[]
    #load all data into a list of dicts
    mb = mailbox.mbox(mbox_file)
    #this is how you access the different features and the text body
    #we need to save everything nicely into one dataset
    for thisemail in mb:
        dmessage = dict(thisemail.items())
        body, attachment = getbodyfromemail(thisemail)
        dmessage['attachment']=attachment
        dmessage['body']= body
        data.append(dmessage)
    print(len(data))
    return data

def separateReplies(data):
    #takes in a list of dicts
    #separates replies and non-replies
    replies=[]
    inits=[]
    for entry in data:
        if 'Reply-To' in entry or 'In-Reply-To' in entry:
            replies.append(entry)
        else:
            inits.append(entry)
    print("len replies:", len(replies))
    print("len initial emails:", len(inits))
    return inits, replies


def parse_attachment(message_part):
    # from https://www.ianlewis.org/en/parsing-email-attachments-python
    content_disposition = message_part.get("Content-Disposition", None)
    if content_disposition:
        dispositions = content_disposition.strip().split(";")
        if bool(content_disposition and dispositions[0].lower() == "attachment"):

            file_data = message_part.get_payload(decode=True)
            attachment = StringIO(file_data)
            attachment.content_type = message_part.get_content_type()
            attachment.size = len(file_data)
            attachment.name = None
            attachment.create_date = None
            attachment.mod_date = None
            attachment.read_date = None

            for param in dispositions[1:]:
                name,value = param.split("=")
                name = name.lower()

                if name == "filename":
                    attachment.name = value
                elif name == "create-date":
                    attachment.create_date = value  #TODO: datetime
                elif name == "modification-date":
                    attachment.mod_date = value #TODO: datetime
                elif name == "read-date":
                    attachment.read_date = value #TODO: datetime
            return attachment

    return None


def seeFeature(data, feature):
    for i in range(10):
            try:
                print(feature,data[i][feature])
            except KeyError:
                print("key error body ")


# #from here on is copied into email classifier
# def performance(y_true, y_pred, metric="accuracy") :
#     """
#     #from ps 6
#     Calculates the performance metric based on the agreement between the 
#     true labels and the predicted labels.
    
#     Parameters
#     --------------------
#         y_true -- numpy array of shape (n,), known labels
#         y_pred -- numpy array of shape (n,), (continuous-valued) predictions
#         metric -- string, option used to select the performance measure
#                   options: 'accuracy', 'f1_score', 'auroc', 'precision',
#                            'sensitivity', 'specificity'        
    
#     Returns
#     --------------------
#         score  -- float, performance score
#     """
#     # map continuous-valued predictions to binary labels
#     y_label = np.sign(y_pred)
#     y_label[y_label==0] = 1 # map points of hyperplane to +1
    
#     ### ========== TODO : START ========== ###
#     # part 2a: compute classifier performance
#     if metric=="accuracy":
#         return metrics.accuracy_score(y_true,y_label)
#     elif metric=="f1_score":
#         return metrics.f1_score(y_true, y_label)
#     elif metric=="auroc":
#         return metrics.roc_auc_score(y_true, y_pred)
#     elif metric=="precision":
#         return metrics.precision_score(y_true, y_label)
#     elif metric=="sensitivity":
#         return metrics.recall_score(y_true, y_label)
#     elif metric=="specificity":
#         cm=metrics.confusion_matrix(y_true,y_label)
#         tn, fp, fn, tp = cm.ravel()
#         return tn/float(tn+fp)
    
#     return 0

# def extract_feature_vectors(data, word_list) :
#     """
   
    
#     Parameters
#     --------------------
#         data            --list of dicts
#         word_list      -- dictionary, (key, value) pairs are (word, index)
    
#     Returns
#     --------------------
#         feature_matrix -- numpy array of shape (n,d)
#                           boolean (0,1) array indicating word presence in a string
#                             n is the number of data points
#                             d is the number of unique words in the text file
#     """
    
#     num_lines = len(data)
#     num_words = len(word_list)
#     feature_matrix = np.zeros((num_lines, num_words))
    
   
       
#     #sorted_word_list=sorted(word_list.items(), key=lambda it: it[1]) # the dictionary sort by its values
#     vector_list=[]
#     for email in data:
#         email_vector=[]
#         email_words=email['tokens']
        

#         for word in word_list:
            
#             if word in email_words:
#                 email_vector.append(1)
               
#             else:
#                 email_vector.append(0)
#         vector_list.append(email_vector)

#     feature_matrix=np.array(vector_list)
    
    
#     return feature_matrix #, sorted_word_list

# def test_performance(clf,x_test,y_test):
#     y_pred=clf.predict(x_test)
#     acc=metrics.accuracy_score(y_test,y_pred)
#     cm=metrics.confusion_matrix(y_test,y_pred)
#     prec=metrics.precision_score(y_test, y_pred)
#     rec=metrics.recall_score(y_test, y_pred)

#     return  prec, rec, acc, cm

# def train_and_test(clf, X_train, y_train, skf, X_test, y_test):
    
#     cv_prec, cv_rec, cv_acc=cv_performance(clf, X_train, y_train, skf)
#     print("CV prec", cv_prec)
#     print("CV rec", cv_rec)
#     print("CV acc", cv_acc)

#     prec, rec, acc, cm= test_performance(clf, X_test, y_test)
#     print("confusion matrix test set", cm)
#     print("test prec", prec)
#     print("test rec",rec)
#     print("test acc", acc)

# def cv_performance(clf, X, y, kf) :
#     """
#     from ps6
#     Splits the data, X and y, into k-folds and runs k-fold cross-validation.
#     Trains classifier on k-1 folds and tests on the remaining fold.
#     Calculates the k-fold cross-validation performance metric for classifier
#     by averaging the performance across folds.
    
#     Parameters
#     --------------------
#         clf    -- classifier (instance of SVC)
#         X      -- numpy array of shape (n,d), feature vectors
#                     n = number of examples
#                     d = number of features
#         y      -- numpy array of shape (n,), binary labels {1,-1}
#         kf     -- model_selection.KFold or model_selection.StratifiedKFold
#         metric -- string, option used to select performance measure
    
#     Returns
#     --------------------
#         score   -- float, average cross-validation performance across k folds
#     """
    
#     precision = []
#     recall=[]
#     accuracy=[]

#     for train, test in kf.split(X, y) :
#         X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
# #        score = performance(y_test, y_pred, metric)
#         acc=metrics.accuracy_score(y_test,y_pred)
#         #cm=metrics.confusion_matrix(y_true,y_pred)
#         prec=metrics.precision_score(y_test, y_pred)
#         rec=metrics.recall_score(y_test, y_pred)

#         # if not np.isnan(score) :
#         #     scores.append(score)

#         precision.append(prec)
#         recall.append(rec)
#         accuracy.append(acc)
        
    
        
#     return np.array(precision).mean(), np.array(recall).mean(), np.array(accuracy).mean()


#['X-Gmail-Labels', 'Delivered-To', 'From', 'Return-Path', 'List-ID', 'Mailing-list', 'X-Gm-Message-State', 'To', 'Message-ID', 'List-Post', 'X-Received', 'X-Google-DKIM-Signature', 'In-Reply-To', 'Date', 'List-Archive', 'body', 'Received', 'Received-SPF', 'Authentication-Results', 'X-Original-Authentication-Results', 'X-BeenThere', 'X-Google-Group-Id', 'Reply-To', 'List-Help', 'MIME-Version', 'Precedence', 'X-Spam-Checked-In-Group', 'X-Original-Sender', 'X-GM-THRID', 'References', 'DKIM-Signature', 'List-Unsubscribe', 'Content-Type', 'Subject']
def main():
    #1 for chat, 0 for dorm
    chat_data=loadData('north-chat.mbox') 
    dorm_data=loadData('north-dorm.mbox')

    inits1,replies1=separateReplies(chat_data)
    inits2, replies2=separateReplies(dorm_data)
   
    
    replies1, dictionary1= clean_text(replies1)
    replies2, dictionary2=clean_text(replies2)
    merged_dictionary=dictionary1+dictionary2
    print("len dictionary", len(merged_dictionary))
    

    feature_matrix1=extract_feature_vectors(replies1,merged_dictionary)
    print(feature_matrix1.shape)
    y1=np.full(feature_matrix1.shape[0], 1)
    print(y1.shape)

        
    feature_matrix2=extract_feature_vectors(replies2, merged_dictionary)
    print(feature_matrix2.shape)
    y2=np.full(feature_matrix2.shape[0], 0)

    y=np.concatenate((y1,y2), axis=0)
    X=np.concatenate((feature_matrix1, feature_matrix2), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=47, shuffle=True)
    np.savez_compressed("reply_only_data" , X=X, y=y)
    skf= StratifiedKFold(n_splits=10, shuffle=True)

    LR=LogisticRegression()
    RF=RandomForestClassifier()
    
    print("log reg")
    train_and_test(LR, X_train, y_train, skf, X_test, y_test)
    print("random forest")
    train_and_test(RF, X_train, y_train, skf, X_test, y_test)
    # with open('iniial_model_BOW_no_replies.pickle', 'wb') as handle:
    #     pickle.dump(clf, handle)



if __name__=="__main__":
    main()