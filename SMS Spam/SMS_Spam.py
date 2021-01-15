# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:53:02 2021

@author: MDTus
"""

#...............................................
# Reading Data

import pandas as pd

dataSet = pd.read_csv('DataSet/spam', sep='\t', names=['label','messages'])
#...............................................


# Head of the database
head = dataSet.head()
print(head)

# Grouping the data to see number of hmm and Spam messages
group = dataSet.groupby('label').describe()
print(group)

#................................................
# Data Processing function

import string as st
from  nltk.corpus import stopwords

def data_pricessing(data):
    """
    Parameters
    ----------
    data : DataFrame
        Remove all Punctuation,
        Remove all Stopwords,
        Remove all Noise
        
    Returns
    -------
    Series
    """   
    
    noise  = [',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤', 'Ã' 'Å', '•', 'Ã³']
    temp1 = [i for i in data if i not in st.punctuation]
    temp1 = ''.join(temp1)
    temp1 = [i for i in temp1.split() if i.lower() not in noise]
    return [i for i in temp1 if i.lower() not in stopwords.words('english')]
    
#................................................


    
    




