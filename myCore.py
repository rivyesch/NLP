# -*- coding: utf-8 -*-
import myDocToTextAndLists
import myExcelReadWrite
import myListsToDataFrames
import time

def add_Simple_Stats(df_to_add_stats_to, per_doc_list, total_docs, valid_docs, num_keywords_to_obtain):
    # Check how many documents a particular word (from the corpus) appears in
    df_to_add_stats_to['numDocs'] = 0
    for doc in per_doc_list:
        for rowIndex, row in df_to_add_stats_to.iterrows():
            word = row[0]
            if word in doc:
                df_to_add_stats_to["numDocs"][rowIndex] = df_to_add_stats_to["numDocs"][rowIndex] + 1

    # Calculating the percentage of all training documents that a particular word (from the corpus) appears in
    df_to_add_stats_to['percentDocs'] = 0
    for rowIndex, row in df_to_add_stats_to.iterrows():
        df_to_add_stats_to['percentDocs'][rowIndex] =  round(((df_to_add_stats_to['numDocs'][rowIndex] / valid_docs)*100), 0)
    
    average_percent = df_to_add_stats_to['percentDocs'].mean()
    
    return df_to_add_stats_to, average_percent     
        
                
#Datasets in the format:
    
    # raw_text       - No processing done
    # all_words_     - For all words, with only some processing done 
    # lemmatized_    - All words + lemmatization & removed bi-grams
    # non_eng_       - Non english words
    
    # _str           - String
    # _dict          - Dictionary, either of strings or dataframes
    # _list          - List
    # _per_doc_list  - List of lists 
    # _df            - Pandas Dataframe containing only words
    # _count_df      - A dataframe containg unique words and their counts
    
    
num_keywords_to_obtain = 20
directory = "Dataset Subset 2"

######################################################################

        #Open Directory, get all files and get rawtext:
            

raw_text_str, raw_text_dict, all_words_list, all_words_per_doc_list, lemmatized_list, lemmatized_per_doc_list, non_eng_list, non_eng_per_doc_list, total_docs, valid_docs = myDocToTextAndLists.text_To_Lists(directory)
print("total docs"+str(total_docs))
print("valid docs"+str(valid_docs))

######################################################################

        #Convert wordLists to dataframes:
            
all_words_df, all_words_count_df, all_words_per_doc_dict = myListsToDataFrames.to_DataFrames(all_words_list, all_words_per_doc_list)
lemmatized_df, lemmatized_count_df, lemmatized_per_doc_dict = myListsToDataFrames.to_DataFrames(lemmatized_list, lemmatized_per_doc_list)
non_eng_df, non_eng_count_df, non_eng_per_doc_dict = myListsToDataFrames.to_DataFrames(non_eng_list, non_eng_per_doc_list)

######################################################################

# %% Save corpus' from excel files
## Legacy code. Intended to use as a faster way to get data back. Kept as the excel file are still useful to inspect.
            
#myExcelReadWrite.dataFrames_To_Excel("ExcelFiles/all_words.xlsx", all_words_df,all_words_count_df,all_words_per_doc_dict)
#myExcelReadWrite.dataFrames_To_Excel("ExcelFiles/lemmatized.xlsx", lemmatized_df,lemmatized_count_df,lemmatized_per_doc_dict)
#myExcelReadWrite.dataFrames_To_Excel("ExcelFiles/non_eng.xlsx", non_eng_df,non_eng_count_df,non_eng_per_doc_dict)




# %%  Bag Of Words
print("BAG OF WORDS:")
bag_Of_Words_Stats_DF, bag_Of_Words_Avg = add_Simple_Stats(non_eng_count_df.head(num_keywords_to_obtain), non_eng_per_doc_list, total_docs, valid_docs, num_keywords_to_obtain)

print(bag_Of_Words_Stats_DF)
print("Mean doc coverage:"+str(bag_Of_Words_Avg))




# %% TextRank solution from a Kaggle post: https://www.kaggle.com/code/john77eipe/textrank-for-keyword-extraction-by-python
# NOTE: Does not work with large datasets. Only use with Dataset Subset 2/3
textrankTic = time.perf_counter()
print("TEXTRANK:")
import kaggleTextRank

kaggle_textRank_All_Keywords_DF = kaggleTextRank.get_Kaggle_TextRank_Keywords(raw_text_str, all_words_df.size)
kaggle_textRank_Non_Eng_Keywords_DF = kaggleTextRank.get_Non_Eng_Keywords(kaggle_textRank_All_Keywords_DF, num_keywords_to_obtain, non_eng_count_df)
kaggle_textRank_Stats_DF, kaggle_textRank_Avg = add_Simple_Stats(kaggle_textRank_Non_Eng_Keywords_DF, non_eng_per_doc_list, total_docs, valid_docs, num_keywords_to_obtain)

print(kaggle_textRank_Stats_DF)
print("Mean doc coverage:"+str(kaggle_textRank_Avg))
textrankToc = time.perf_counter()



# %% LDA and HDP with Gensim: https://stackoverflow.com/questions/45861220/extract-most-important-keywords-from-a-set-of-documents
ldaTic = time.perf_counter()
num_topics = 20 #Determines how many topics LDA uses. 1=BagOfWords. Use ~20
print("GENSIM LDA (with "+str(num_topics)+" topics, "+str(num_keywords_to_obtain)+" keywords):")
import gensimLDA

gensim_keyword_df = gensimLDA.generate_Keyword_DF(non_eng_per_doc_list, non_eng_count_df, num_keywords_to_obtain, num_topics)
gensim_Stats_DF, gensim_Avg = add_Simple_Stats(gensim_keyword_df, non_eng_per_doc_list, total_docs, valid_docs, num_keywords_to_obtain)

print(gensim_Stats_DF)
print("Mean doc coverage:"+str(gensim_Avg))
ldaToc = time.perf_counter()



# %% Produce graphs / stats...

#print(f"textrank time: {textrankToc-textrankTic:0.4f} seconds")
#print(f"lda time: {ldaToc-ldaTic:0.4f} seconds")

import matplotlib.pyplot as plt

# %% Experiment 1 : LDA num of topics to use

# LDANumTopics = [26.75, 22, 20.51, 19.97, 19.95, 19.84, 19.84]
# x = [1, 5, 10, 15, 20, 25, 30] 

# plt.plot(x, LDANumTopics)
# plt.title("LDA - finding the best number of topics to use")
# plt.xlabel("Number of Topics")
# plt.ylabel("Mean document coverage of top 20 Keywords")
# plt.annotate("Curve levels off",xy=(20,20),xytext=(20,22),arrowprops={"width":4,"headwidth":15,'headlength':15}, horizontalalignment='center',fontsize=10)
# plt.show()

# %% Experiment 2: Num input documents vs mean coverage 

# BagSet = [23.1,	19.5,	17.2,	18.15,	17.4,	17.25,	17.1,	18.3,	17.45,	18.6]
# TextSet = [19.9,	21.5,	16.7,	16.8,	15.95,	15.3,	16.45,	17.85,	16.95,	17.7]
# LDASet = [21.61,	15.7,	11.69,  13.21, 	12.81,	13.6,	13.42,	15.19,	13.93,	13.53]

# x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# plt.plot(x, BagSet, label="Bag Of Words")
# plt.plot(x, TextSet, label="TextRank")
# plt.plot(x, LDASet, label="LDA")
# plt.xlabel("Number of valid input documents")
# plt.ylabel("Mean document coverage of top 20 Keywords")
# plt.legend()
# plt.show()

# %% Experiment 3: Num input documents vs time to process

# TextTimes = [21.03,	49.13,	69.04,	85.18,	148.60,	172.59,	204.94,	233.64,	287.16,	405.29]
# LDATimes = [0.21,	0.35,	0.49,	0.64,	0.80,	0.89,	1.00,	1.19,	1.32,	1.51]

# x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# plt.plot(x, TextTimes, label="TextRank")
# plt.plot(x, LDATimes, label="LDA")
# plt.xlabel("Number of valid input documents")
# plt.ylabel("Time to execute")
# plt.legend()
# plt.show()