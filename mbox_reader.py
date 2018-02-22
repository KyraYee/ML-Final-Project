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
from nltk.stem.snowball import SnowballStemmer


#--------------------------------------------------------------------------------------------------------------------------------
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
                       

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                

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

#----------------------------------------------------------------------------------------------------------------------------


def makeStopWordList():
    """
    Gives a stopword list which is the standard NLTK stopword list with some specialized additions for emails
    """
    stop_words=stopwords.words('english')
    extra=[".", ",", "<", ">", "@", "+", "=", "subscribe", "unsubscribe", ""]
    stop_words=stop_words+extra
    return stop_words

def clean_text(data):
    """
    Clean the text of the emails and perform tokenization
    Parameters
    --------------------
       data -- list of dictionaries, where each entry in the list represents one email. The dictionary keys are different data feature types for the emails
    
    Returns
    --------------------
        data -- list of dictionaries, with an added feature "tokens", with the tokenized version of the email body
        dictionary -- a list of all words in the vocabulary of the corpus
    """
    dictionary=[]
    stop_words=makeStopWordList()
    #takes in a list of dicts
    for entry in data:
        body=entry['body']
        if body !=None:
            tokens = nltk.word_tokenize(body)
            stemmer = SnowballStemmer("english")
            tokens=[stemmer.stem(word) for word in tokens if word not in stop_words and len(word)<30]  #tokens are unicode
            for word in tokens:
                #word is of type unicode
                try:
                    clean_word=word.encode('ascii', 'ignore')
                    if clean_word not in dictionary and "@" not in clean_word :
                        dictionary.append(clean_word)
                except UnicodeDecodeError:
                    print("unicode error in text cleaning")
                    print(word)
            entry['tokens']=tokens
        else:
            print("no tokens")
            entry['tokens']=None
    dictionary =[word for word in dictionary if dictionary[word]>20] #each word must appear at least 20 times in the corpus to be included in the vocabulary

    return data , dictionary




def loadData(mbox_file):
    """
    load the mbox data
    --------------------
       mbox_file -- string, the name of the mbox file to open
    
    Returns
    --------------------
        data -- a list of dictionaries, where each entry in the list corresponds to the data from a single email

    """
    data=[]
    #load all data into a list of dicts
    mb = mailbox.mbox(mbox_file)
    #this is how you access the different features and the text body
    for thisemail in mb:
        dmessage = dict(thisemail.items())
        body, attachment = getbodyfromemail(thisemail)
        dmessage['attachment']=attachment
        dmessage['body']= body
        data.append(dmessage)
    return data

def separateReplies(data):
     """
    load the mbox data
    --------------------
       data -- list of dicts, with the loaded mbox data
    
    Returns
    --------------------
        inits   -- a list of dictionaries, where each entry in the list corresponds to the data from a single email for initial emails
        replies -- a list of dictionaries, where each entry in the list corresponds to the data from a single email for emails that were replies

    """
    replies=[]
    inits=[]
    for entry in data:
        if 'Reply-To' in entry or 'In-Reply-To' in entry:
            replies.append(entry)
        else:
            inits.append(entry)
    print("number of  replies:", len(replies))
    print("number of initial emails:", len(inits))
    return inits, replies

 # from https://www.ianlewis.org/en/parsing-email-attachments-python
#---------------------------------------------------------------------------------------
def parse_attachment(message_part):
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

#---------------------------------------------------------------------------------------

def seeFeature(data, feature):
    """
    look at a given feature
    """
    for i in range(10):
            try:
                print(feature,data[i][feature])
            except KeyError:
                print("key error body ")

def doWtoNum(doW):
    """ 
    encode each day of the week string as a number
    """
    if doW == 'Sun':
        return 0
    elif doW == 'Mon': 
        return 1
    elif doW == 'Tue':
        return 2
    elif doW == 'Wed':
        return 3
    elif doW == 'Thu':
        return 4
    elif doW == 'Fri':
        return 5
    elif doW == 'Sat':
        return 6
    else:
        return 7



#This is all of the raw mbox features, for reference
#['X-Gmail-Labels', 'Delivered-To', 'From', 'Return-Path', 'List-ID', 'Mailing-list', 'X-Gm-Message-State',
# 'To', 'Message-ID', 'List-Post', 'X-Received', 'X-Google-DKIM-Signature', 'In-Reply-To', 'Date', 'List-Archive',
# 'body', 'Received', 'Received-SPF', 'Authentication-Results', 'X-Original-Authentication-Results', 'X-BeenThere', 
#'X-Google-Group-Id', 'Reply-To', 'List-Help', 'MIME-Version', 'Precedence', 'X-Spam-Checked-In-Group', 
#'X-Original-Sender', 'X-GM-THRID', 'References', 'DKIM-Signature', 'List-Unsubscribe', 'Content-Type', 'Subject']
