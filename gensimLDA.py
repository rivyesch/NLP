# -*- coding: utf-8 -*-

import gensim
from gensim.models import LdaModel
from gensim import corpora
import pandas as pd

# Ref: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#20topicdistributionacrossdocuments

def order_subset_by_coherence(dirichlet_model, bow_corpus, num_topics, num_keywords):

    # Function to derive the best topics in order based on their average coherence across the corpus
    # Ref: https://stackoverflow.com/a/64284694
    
    
    shown_topics = dirichlet_model.show_topics(num_topics=num_topics, 
                                                   num_words=num_keywords,
                                                   formatted=False)
   

    # Ordered lists for the most important words per topic
    model_topics = [[word[0] for word in topic[1]] for topic in shown_topics]
    topic_corpus = dirichlet_model.__getitem__(bow=bow_corpus, eps=0) # cutoff probability to 0 

    topics_per_response = [response for response in topic_corpus]
    flat_topic_coherences = [item for sublist in topics_per_response for item in sublist]

    significant_topics = list(set([t_c[0] for t_c in flat_topic_coherences])) # those that appear
    topic_averages = [sum([t_c[1] for t_c in flat_topic_coherences if t_c[0] == topic_num]) / len(bow_corpus) \
                      for topic_num in significant_topics]

    # Average coherence of each topic to the whole corpus is found
    topic_indexes_by_avg_coherence = [tup[0] for tup in sorted(enumerate(topic_averages), key=lambda i:i[1])[::-1]]

    significant_topics_by_avg_coherence = [significant_topics[i] for i in topic_indexes_by_avg_coherence]

    # Topics are ordered based on this average coherence along with the corresponding averages
    ordered_topics = [model_topics[i] for i in significant_topics_by_avg_coherence][:num_topics] # limit for HDP

    ordered_topic_averages = [topic_averages[i] for i in topic_indexes_by_avg_coherence][:num_topics] # limit for HDP
    ordered_topic_averages = [a/sum(ordered_topic_averages) for a in ordered_topic_averages] # normalize HDP values

    return ordered_topics, ordered_topic_averages


def generate_N_Keywords(a_per_doc_list, num_topics, num_keywords):

    # This function does the following
    # Creates the LDA model
    # Extracts the top keywords which are essentially the topics with the highest average coherence across the corpus

    # Ref: https://stackoverflow.com/a/64284694
    
    corpus = a_per_doc_list
  
    # Creates a dirichlet dictionary that maps words in corpus to indexes
    dirichlet_dict = corpora.Dictionary(corpus)
    # Tokens within corpus are replaced by their indexes
    bow_corpus = [dirichlet_dict.doc2bow(text) for text in corpus]
    
    # Creating the LDA model
    dirichlet_model = LdaModel(corpus=bow_corpus,
                           id2word=dirichlet_dict,
                           num_topics=num_topics,
                           update_every=1,
                           chunksize=len(bow_corpus),
                           passes=20,
                           alpha='auto')
    
    ordered_topics, ordered_topic_averages = order_subset_by_coherence(dirichlet_model, bow_corpus, num_topics, num_keywords)
    
    
    keywords = []
    for i in range(num_topics):
        # Find the number of indexes to select, which can later be extended if the word has already been selected
        selection_indexes = list(range(int(round(num_keywords * ordered_topic_averages[i]))))
        if selection_indexes == [] and len(keywords) < num_keywords: 
            # Fix potential rounding error by giving this topic one selection
            selection_indexes = [0]
                  
        for s_i in selection_indexes:
            if ordered_topics[i][s_i] not in keywords:
                keywords.append(ordered_topics[i][s_i])
            else:
                selection_indexes.append(selection_indexes[-1] + 1)
    
    # Fix for if too many were selected
    keywords = keywords[:num_keywords]
    
    return keywords



def generate_Keyword_DF(a_per_doc_list, a_count_df, num_keywords, num_topics):

    # Function to generate the top keywords and store it in a dataframe. Also, calls another function to perform a simple count
    # of the frequency of each keywords in the corpus and appends it in the dataframe accordingly

    keywords = generate_N_Keywords(a_per_doc_list, num_topics, num_keywords)
    
    keyword_DF_temp = pd.DataFrame(keywords, columns=['words'])    
    keyword_DF = keyword_DF_temp.merge(a_count_df, how='left' )
     
    return keyword_DF
    