import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import words
from english_words import english_words_lower_set


# Punctuation and symbols to be removed from the corpus
remove_single_punctuation = ['(',')',';',':','[',']',',','/', '(', ')', '*', '?', "'", '-', "''", "``", "&", "%", "-", '.', '’', '•', '–', '‘', '”', '“','●','·','│','']
punctuation_to_remove = "#$%&'()*+,/:;<=>?@[\]^_`{|}~’•–‘”“●" # -. not included

# list of english stopwords which are considered to be insignificant in NLP tasks from NLTK
stop_words = stopwords.words('english')
# including additional english stopwords. Some used for testing purposes 
more_stop_words = ['name','website','metre','postcode','email','yes','no','address',
                     'plan','please','planning','line','if','are','e.g','we','also', 
                     'the','it','you','is','would','for','etc','where','with','x','it',
                     'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# combining all stop words to create a more complete list of english stopwords to filter out from the corpus
stop_words = stop_words+more_stop_words

# list of english words from NLTK
NLTKWordsList = set(words.words())
# combining the list of english words from NLTK with a list of english words from the package english_words to make a more complete list
combinedDict = english_words_lower_set.union(NLTKWordsList)