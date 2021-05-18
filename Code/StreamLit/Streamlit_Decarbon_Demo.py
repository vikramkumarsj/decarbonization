# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:17:26 2020

@author: Vikram.V.Kumar
"""

# Import Libraries

import streamlit as st
import time
import spacy
import pandas as pd
import numpy as np
import glob
from PIL import Image
from pytesseract import *
pd.set_option('display.max_colwidth', -1)

import pdf2image
import docx2txt

image = Image.open('/app/decarbonization/Data/Shell_Image.jpg')
st.image(image)
st.title("Decarbon related keywords found in the Technical document") 

# streamlit_app.py
import spacy_streamlit

 
# In[12]:

df = pd.DataFrame(columns=['alltext'])   
contents  = "ML as a powerful emerging tool for carbon capture applications as well. ML methods have enabled researchers to design, test, and improve various aspects of the process that are computationally (using theoretical calculations) or experimentally time consuming and expensive. The use of ML for carbon capture processes is still emerging, and most investigations have been conducted only in the past few years. The primary goal in implementation of ML for carbon capture is to design and execute process schemes to effectively separate CO2 from a gas mixture (e.g., CO2/N2/O2) with a minimum energy penalty and cost. ML methods have been implemented successfully in both absorption- and adsorption-based processes, at the molecular and the process level, to overcome challenges that these approaches are currently facing. In this perspective, we discuss how ML has been adapted to predict the thermodynamic properties of CO2-absorbent chemistry, such as the solubility, to facilitate the discovery of alternative absorbents"
contents = contents.lower()

df = df.append({'alltext': contents}, ignore_index=True)

final_data = pd.DataFrame([])
for x in range(0,8):
    start = (x)*100000
    end = (x+1)*100000
#    print(start,end)
    df_subset = df[start:end]
#    df_subset = remove_stop_words(df_subset)
    final_data = final_data.append(df_subset, ignore_index=True)
#final_data = pd.DataFrame(final_data)
df = final_data
#print(df.shape)
#df.head()

st.write(df)



# # Apply the model on cleansed data

# In[19]:
#nlp = spacy.load('C:\\Users\\Vikram.V.Kumar\\Desktop\\\Work\\Projects\\GLIMP\\code\\model\myNlp_ensemble')

from spacy_streamlit import visualize_textcat
from spacy_streamlit import visualize_tokens
from spacy_streamlit import load_model

entity_list = pd.DataFrame()
spacy_model = st.sidebar.selectbox("Select NLP Model to be Applied", ["Custom NER", "Custom Text Categorizer", "OOTB NER" ])

if spacy_model == "Custom NER":
        
#        nlp = load_model("en_core_web_lg")
        nlp.vocab.vectors = spacy.vocab.Vectors(data=med_vec.syn0, keys=med_vec.vocab.keys())
        nlp = spacy.load("/app/decarbonization/Model/Decarbon_NER_Model_OOTB")
        st.write("Model Imported")


else:
    st.write("No Model")


   
















