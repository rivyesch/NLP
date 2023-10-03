# -*- coding: utf-8 -*-

import pandas as pd
import openpyxl

# Legacy code. Original plan was to use this to make building / retrieving datasets faster. Didn't end up utilising this.

def get_Dict_From_Excel(filename):

    aPandasExcelFile = pd.ExcelFile(filename)
    
    docsDict = {}
    
    for sheetName in aPandasExcelFile.sheet_names:  
        thisDocsDF = pd.read_excel(filename, sheet_name=sheetName)  
        docsDict[sheetName] = thisDocsDF
            
    return docsDict
        

    
def dataFrames_To_Excel(docName, data_df, data_counts_df, data_per_doc_dict):
    newFile = openpyxl.Workbook()
    newFile.save(filename=docName)
    
    with pd.ExcelWriter(docName, engine="openpyxl", mode="a") as writer:
        data_df.to_excel(writer, sheet_name='data_df', index=False) 
        data_counts_df.to_excel(writer, sheet_name='data_counts_df', index=False)
        
    DF_Or_Count = True
    doc_index = 0
    for element in data_per_doc_dict:
        #print(data_per_doc_dict[element])
        if DF_Or_Count:
            with pd.ExcelWriter(docName, engine="openpyxl", mode="a") as writer:
                data_per_doc_dict[element].to_excel(writer, sheet_name='doc'+str(doc_index), index=False)
            DF_Or_Count = False
        else:
            with pd.ExcelWriter(docName, engine="openpyxl", mode="a") as writer:
                data_per_doc_dict[element].to_excel(writer, sheet_name='doc'+str(doc_index)+'counts', index=False)
            DF_Or_Count = True
            doc_index += 1
    return True
    
    