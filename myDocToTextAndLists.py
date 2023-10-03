from nltk.tokenize import word_tokenize
import re 
import fitz
import os

import myConstants
import myEnglishChecker


def text_To_Lists(directoryName):
    # This function imports the raw text from each pdf file into Python as a list

    raw_text_str = ""
    raw_text_dict = {}
    
    all_words_list = []
    all_words_per_doc_list = []

    lemmatized_list = []
    lemmatized_per_doc_list = []

    non_eng_list = []
    non_eng_per_doc_list = []

    valid_docs = 0
    total_docs = 0
    
    for root, dirs, files in os.walk(directoryName):
        for file in files:
            f = os.path.join(root, file)
            print(f)
    
            rawText = ''
            try:
                with fitz.open(f) as doc:
                    for page in doc:
                        rawText += page.get_text()
            except:
                print("a file could not be opened:"+f)        
                    
            #Turn the rawtext into a list, removing all useless information
            cleaner_all_words_list = clean_Raw_Text(rawText)    
                    
            #Lemmatize & remove Bi-grams it for easier processing
            lemmatized_words = myEnglishChecker.lemmatize_Clean_Text_List(cleaner_all_words_list)
            
            #Find non english terms
            this_doc_non_eng_list = myEnglishChecker.find_Non_English_Words(lemmatized_words)
            
            
            raw_text_str += rawText
            raw_text_dict["doc"+str(total_docs)] = rawText
            
            all_words_list.extend(cleaner_all_words_list)
            all_words_per_doc_list.append(cleaner_all_words_list)
            lemmatized_list.extend(lemmatized_words)
            lemmatized_per_doc_list.append(lemmatized_words)
            non_eng_list.extend(this_doc_non_eng_list)
            non_eng_per_doc_list.append(this_doc_non_eng_list)
            
            
            total_docs += 1
            if cleaner_all_words_list != []:
                valid_docs += 1
    return raw_text_str, raw_text_dict, all_words_list, all_words_per_doc_list, lemmatized_list, lemmatized_per_doc_list, non_eng_list, non_eng_per_doc_list, total_docs, valid_docs


def clean_Raw_Text(rawText):
    # This function takes raw sentences and 'cleans it', by performing the following:
    # converts all text to lowercase
    # removes english stopwords
    # removes punctuation (except in 'bi-grams' or in initials 'e.g.')
    # removes standalone numbers (such as 5123 but will not remove P60)
    # removes floats
    # removes blank spaces

    # NLTK tokenizer to divide strings of text into list of substrings that contain each individual words, punctuation, etc.
    tokens = word_tokenize(rawText)

    # splits yes/no and dd/mm/yyyy which wasn't split up by the tokenization
    tokens = [item.split('/') for item in tokens] #splits yes/no dd/mm/yyyy
    tokens = [item for l in tokens for item in l]

    # extracts those elements in "tokens" that are not stop words, punctuations and numbers (filter step)
    all_words_list = [word.lower() for word in tokens if (word not in myConstants.stop_words) and (not word in myConstants.remove_single_punctuation) and (not word.isdigit())]
    # removes symbols
    all_words_list = [word.translate(str.maketrans('','',myConstants.punctuation_to_remove)) for word in all_words_list] #Remove punctuation
    all_words_list = [word.strip(".-") for word in all_words_list] #Remove .- from end of word
    # removes standalone numbers
    all_words_list = [word for word in all_words_list if word.isnumeric() != True] 
    all_words_list = [word for word in all_words_list if not re.match('^[0-9\.]*$',word)] #^^ Catch floats
    # removes empty spaces (must come last as previous step causes empty spaces)
    all_words_list = [word for word in all_words_list if word != ""] #Remove empty spaces (MUST COME LAST AS PREVIOUS STEPS CAUSE EMPTY SPACES)
 
    return all_words_list

