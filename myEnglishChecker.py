# -*- coding: utf-8 -*-

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')

import myConstants

lem = WordNetLemmatizer()

def find_Non_English_Words(my_list):
    # Function to find non-english words by comparing words from cleaned pre-processed corpus with a comprehensive list of english words

    eng = []
    not_eng = []
    
    for word in my_list:
        if word in myConstants.combinedDict:
            eng.append(word)
        else:
            not_eng.append(word)

    return not_eng


def get_Wordnet_Pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)   

def check_Bigrams(word):
    # This function does the following:
    # Identify bi-grams by searching for presence of '-' in words in corpus
    # Checks if both words are english
    # If both words are English, then the bi-gram is split into the two individual words
    # If either one of the words are not English, then the bi-gram is retained (and not split)

    bigram_Marker = '-'
    
    if bigram_Marker in word:
        seperatedWord = word.split(bigram_Marker)
        
        lemLeft = lem.lemmatize(seperatedWord[0], get_Wordnet_Pos(seperatedWord[0]))
        lemRight = lem.lemmatize(seperatedWord[1], get_Wordnet_Pos(seperatedWord[1]))
        
        if (lemLeft.lower() in myConstants.combinedDict and lemRight.lower() in myConstants.combinedDict) or (lemLeft.isdigit() and lemRight.isdigit()):
            return lemLeft
        else:
            return word
    else:        
        return word 

def lemmatize_Clean_Text_List(textList):
    # This function does the following:
    # Remove English bi-grams while keeping non-English ones
    # Lemmatize words in corpus

    removedBigrams = [check_Bigrams(word) for word in textList]
    myLemmatizedList = [lem.lemmatize(word, get_Wordnet_Pos(word)) for word in removedBigrams]
    # Remove any stop words post-lemmatization
    myLemmatizedList = [word for word in myLemmatizedList if word not in myConstants.stop_words]

    return myLemmatizedList

