# -*- coding: utf-8 -*-
#https://www.kaggle.com/code/john77eipe/textrank-for-keyword-extraction-by-python

import numpy as np
import pandas as pd 


from collections import OrderedDict
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')

class TextRankForKeyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        
        keywords = []
        
        for i, (key, value) in enumerate(node_weight.items()):
            #print(key + ' - ' + str(value))
            keywords.append(key)
            if i > number:
                break
        return keywords
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False):
    
        
        # Pare text by spaCy
        doc = nlp(text)
        
        #for sent in doc.sents:
        #    for token in sent:
        #        True
        #        #print(token.text)
        for sent in doc.sents:
            displacy.render(sent, style="dep")    
        for sent in doc.sents:
            displacy.render(sent, style="ent")
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        #print(token_pairs)
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initialization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight
        
        


def get_Kaggle_TextRank_Keywords(text, num_keywords_to_obtain):
    import warnings
    warnings.filterwarnings('ignore')

    tr4w = TextRankForKeyword()
    tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
    keywords = tr4w.get_keywords(num_keywords_to_obtain)
    
    return keywords
    
def get_Non_Eng_Keywords(keywords, num_keywords_to_obtain, non_eng_count_df):
    
    #Suboptimal... need to refactor myEnglishChecker / myDocToTextAndLists classes to accomodate this.
    import myConstants
    import re
    keywords_list = [word.lower() for word in keywords if (word not in myConstants.stop_words) and (not word in myConstants.remove_single_punctuation) and (not word.isdigit())]
    keywords_list = [word.translate(str.maketrans('','',myConstants.punctuation_to_remove)) for word in keywords_list] #Remove punctuation 
    keywords_list = [word.strip(".-") for word in keywords_list] #Remove .- from end of word
    keywords_list = [word for word in keywords_list if word.isnumeric() != True] #Remove standalone numbers
    keywords_list = [word for word in keywords_list if not re.match('^[0-9\.]*$',word)] #^^ Catch floats
    keywords_list = [word for word in keywords_list if word != ""] #Remove empty spaces (MUST COME LAST AS PREVIOUS STEPS CAUSE EMPTY SPACES)
    
    import myEnglishChecker
    keywords_Lemmatized = myEnglishChecker.lemmatize_Clean_Text_List(keywords_list)
    non_eng_keywords = myEnglishChecker.find_Non_English_Words(keywords_Lemmatized)
    
    non_eng_keywords = list(dict.fromkeys(non_eng_keywords)) #Removes duplicate words
    
    keyword_DF_temp = pd.DataFrame(non_eng_keywords, columns=['words'])    
    keyword_DF = keyword_DF_temp.merge(non_eng_count_df, how='left' )
    keyword_DF = keyword_DF.dropna().reset_index(drop=True)

    top_Keyword_DF = keyword_DF.head(num_keywords_to_obtain)
    top_Keyword_DF['numFound'] = top_Keyword_DF['numFound'].astype(int) 
    #Reason, left merge may have missing values.
    #Only issue on small datasets

    return top_Keyword_DF