# -*- coding: utf-8 -*-
import pandas as pd

def to_DataFrames(a_list, a_per_doc_list):
    # This function creates a dataframe with two columns - unique words and the corresponding count of each word

    # Dataframe of all words
    a_df = pd.DataFrame(a_list, columns=['words'])
    
    # Count all unique words
    a_count_df = a_df.value_counts()
    a_count_df = a_count_df.reset_index()
    a_count_df.columns = ['words', 'numFound']    
    
    # Generate per doc dict
    a_doc_dict = {}
    
    doc_index = 0
    for a_doc_list in a_per_doc_list:
        
        a_doc_df = pd.DataFrame(a_doc_list, columns=['words'])
    
        a_doc_count_df = a_doc_df.value_counts()
        a_doc_count_df = a_doc_count_df.reset_index()
        a_doc_count_df.columns = ['words', 'numFound']
    
        if a_doc_df.empty != True:
            a_doc_dict["doc"+str(doc_index)] = a_doc_df
            a_doc_dict["doc"+str(doc_index)+"count"] = a_doc_count_df
        
        doc_index += 1
    
    return a_df, a_count_df, a_doc_dict
