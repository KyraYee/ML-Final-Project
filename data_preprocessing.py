from nltk.stem.snowball import SnowballStemmer
import numpy as np
from mbox_reader import *

def getWordCounts(email_tokens):
    """
    get the counts of each word in the corpus
    
    Parameters
    --------------------
        email_tokens -- list of strings, the emails tokenized      
    
    Returns
    --------------------
        counts -- a dictionary where keys are words and values are counts of words in the corpus
    """
    counts={}
    try:
        stemmer = SnowballStemmer("english")
        email_words= [stemmer.stem(word) for word in email_tokens]
        for token in email_words:
            if token in counts:
                counts[token]+=1
            else:
                counts[token]=1
    except UnicodeDecodeError:
        print("token list unicode error")

    return counts



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
    
    
    return feature_matrix 


def generateBOWData():
    """
    Processes mbox data into feature vectors 
    
    """
     #1 for chat, 0 for dorm
    chat_data=loadData('north-chat.mbox') 
    dorm_data=loadData('north-dorm.mbox')

    inits1,replies1=separateReplies(chat_data)
    inits2, replies2=separateReplies(dorm_data)

    inits1=[x for x in inits1 if x['body']!=None]
    inits2=[x for x in inits2 if x['body']!=None]

    print("len chat inits", len(inits1))
    print("len dorm inits", len(inits2))

    y1=np.full(len(inits1),1)
    y2=np.full(len(inits2),0)
    y=np.concatenate((y1,y2), axis=0)

    X_unprocessed= inits1+inits2

    assert len(X_unprocessed)==y.shape[0]
    print("type X", type(X_unprocessed))
    print("type X[0]", type(X_unprocessed[0]))

    X, dictionary= clean_text(X_unprocessed)
    print("len dictionary", len(dictionary))
    X=extract_feature_vectors(X, dictionary)
    print("training data shape", X.shape)

    np.savez_compressed("init_only_data" , X=X, y=y, dictionary=dictionary)

def generateeNonalphanum(load_file, save_file):
     """
     takes in a previously generated dataset and adds the number of nonalphanumeric characters as a datafeature
    
    Parameters
    --------------------
        load_file -- string, filename of the previously generated dataset
        save_file -- string, the name of the new dataset with nonalphanum feature
    
    """
    a=np.load(load_file)
    emails=a['raw_X']
    y=a['y']
    X=a['X']
    dictionary=a['dictionary']
    assert len(emails)==len(X)

    newX=[]
    for index in range(len(X)):
        email=emails[index]
        newX.append(np.append(nonAlphaNumericCount(email), X[index])) #append the new feature to the beginning of the feauture list
    newX=np.array(newX)
    print(newX.shape)
    np.savez_compressed(save_file , X=newX, y=y, raw_X=emails, dictionary=dictionary)

def generateAttachment(load_file, save_file):
     """
     takes in a previously generated dataset and adds whether or not there was an attachment as a datafeature
    
    Parameters
    --------------------
        load_file -- string, filename of the previously generated dataset
        save_file -- string, the name of the new dataset with attachment feature
    
    """
    a=np.load(load_file)
    emails=a['raw_X']
    y=a['y']
    X=a['X']
    dictionary=a['dictionary']
   
    assert len(emails)==len(X)

    newX=[]
    for index in range(len(X)):
        email=emails[index]

        if email['attachment']==None:
            newX.append(np.append(0,X[index]))
        else:
            newX.append(np.append(1,X[index]))
    newX=np.array(newX)
    print(newX.shape)
    np.savez_compressed(save_file , X=newX, y=y, raw_X=emails, dictionary=dictionary)

def generateTime(load_file, save_file):
     """
     takes in a previously generated dataset and adds the day of week and the time sent
    
    Parameters
    --------------------
        load_file -- string, filename of the previously generated dataset
        save_file -- string, the name of the new dataset with attachment feature
    
    """
    a=np.load(load_file)
    raw_X=a['raw_X']
    y=a['y']
    X=a['X']
    dictionary=a['dictionary']

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
    
    np.savez_compressed(save_file , X=Xfinal, y=y, raw_X=raw_X, dictionary=dictionary)

def capCount(message):
    """
    Parameters
    --------------------
        message -- string
    
    Returns
    --------------------
        capCount -- number of capital letters
    """
    if message is None:
        return 0
    capCount = sum(1 for c in message if c.isupper())
    return capCount

def asteriskCount(message):
    """
    Parameters
    --------------------
        message -- string
    
    Returns
    --------------------
        asteriskCount -- number of asterisks
    """
    if message is None:
        return 0
    asteriskCount = sum(1 for c in message if c=='*')
    return asteriskCount

def nonAlphaNumeric(c):
     """
    Parameters
    --------------------
       c - character
    
    Returns
    --------------------
        whether or not c is nonalphanumeric
    """
    if c.isalpha():
        return False
    if c.isdigit():
        return False
    else:
        return True

def nonAlphaNumericCount(message):
    """
    Parameters
    --------------------
        message -- string
    
    Returns
    --------------------
        nonAlphaNumericCount -- number of nonalphanumeric characters
    """
    if message is None:
        return 0
    nonAlphaNumericCount = sum(1 for c in message if nonAlphaNumeric(c))
    return nonAlphaNumericCount


    
