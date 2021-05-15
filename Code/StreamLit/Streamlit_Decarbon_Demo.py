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
from pytesseract import *

# In[2]:


import os
import re



# In[4]:


def pdftopil(PDF_PATH, DPI, OUTPUT_FOLDER, FIRST_PAGE, LAST_PAGE, FORMAT, THREAD_COUNT, USERPWD):
    #This method reads a pdf and converts it into a sequence of images
    #PDF_PATH sets the path to the PDF file
    #dpi parameter assists in adjusting the resolution of the image
    #output_folder parameter sets the path to the folder to which the PIL images can be stored (optional)
    #first_page parameter allows you to set a first page to be processed by pdftoppm 
    #last_page parameter allows you to set a last page to be processed by pdftoppm
    #fmt parameter allows to set the format of pdftoppm conversion (PpmImageFile, TIFF)
    #thread_count parameter allows you to set how many thread will be used for conversion.
    #userpw parameter allows you to set a password to unlock the converted PDF
    #use_cropbox parameter allows you to use the crop box instead of the media box when converting
    #strict parameter allows you to catch pdftoppm syntax error with a custom type PDFSyntaxError

#    start_time = time.time()
    pil_images = pdf2image.convert_from_path(PDF_PATH, dpi=DPI, output_folder=OUTPUT_FOLDER, first_page=FIRST_PAGE, last_page=LAST_PAGE, fmt=FORMAT, thread_count=THREAD_COUNT, userpw=USERPWD)
#    print ("Time taken : " + str(time.time() - start_time))
    return pil_images


# In[5]:


def save_images(pil_images):
    #This method helps in converting the images in PIL Image file format to the required image format
    index = 1
    for image in pil_images:
#        image.save("page_" + str(index) + ".jpg")
        image.save('/app/decarbonization/Data/Images/'+"page_" + str(index) + ".jpg")
        index += 1


# In[6]:


def extract_text_from_image(image_path):
    value=Image.open(image_path)
    pytesseract.tesseract_cmd = '/app/decarbonization/Data/Tesseract-OCR/tesseract'
    text = pytesseract.image_to_string(value)
    return text


# In[7]:


def deleteTempImage(filePath):
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file ", filePath)


# In[8]:


def deleteContent(pfile):
    fn=pfile.name 
    pfile.close()
    return open(fn,'w')


# In[9]:


def Process_pdf(PDF_PATH):
#    PDF_PATH = "C:\\Users\\Vikram.V.Kumar\\Desktop\\Auto Tagging\\Data\\CSO\\India\\Structural Modifications to Existing Deck.pdf"
    PDF_PATH = PDF_PATH
    DPI = 200
    OUTPUT_FOLDER = None
    FIRST_PAGE = None
    LAST_PAGE = None
    FORMAT = 'jpg'
    THREAD_COUNT = 1
    USERPWD = None
    USE_CROPBOX = False
    STRICT = False
    
    pil_images = pdftopil(PDF_PATH, DPI, OUTPUT_FOLDER, FIRST_PAGE, LAST_PAGE, FORMAT, THREAD_COUNT, USERPWD)
    save_images(pil_images)
    
    f=open("/app/decarbonization/Data/Full_Text/extractedImageText.txt", "w", encoding="utf-8")
    imagePath = "/app/decarbonization/Data/Full_Text/Images/*.jpg"
    files = glob.glob(imagePath)
    PDF_PATH_BKP = PDF_PATH
    PDF_PATH = PDF_PATH.replace('\\',' ')
    f.write(PDF_PATH+' ')
    
    for name in files:
        imageText = extract_text_from_image(name)
        f.write(imageText)
        deleteTempImage(name)

#    for name in files:
#        try:
#            imageText = extract_text_from_image(name)
#            f.write(imageText)
#            deleteTempImage(name)
#        except IOError as exc:
#            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
#                raise # Propagate other kinds of IOError.
                
    f=open("/app/decarbonization/Data/Full_Text/extractedImageText.txt", "r", encoding="utf-8")
    try:
        contents =f.read()
    finally:
        deleteContent(f)
    
    contents = contents.lower()

        
        
    return contents     
    
       
    
    

        
    


# In[10]:


def Process_Doc(PDF_PATH):
    
    contents = docx2txt.process(PDF_PATH)    
    contents = contents.lower()
#    print(contents)
 
    return contents   



def clean_it(text):
    text = re.sub('; ', ' ', text)
    text = re.sub(', ', ' ', text)
    text = re.sub('\. ',' ', text)
#    text = re.sub(',', '_', text)
    
    return text

def remove_stop_words(df):
    import spacy
    from spacy.lang.en import English
    
#    import modin.pandas as pd
    df = pd.DataFrame(df)
    
    nlp_clean = English()
    from spacy.lang.en.stop_words import STOP_WORDS
    
    for index, row in df.iterrows():
        my_doc = nlp_clean(row['alltext'])
        token_list = []
        for token in my_doc:
            token_list.append(token.text)
        filtered_sentence =[] 
        for word in token_list:
            lexeme = nlp_clean.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)
                
        newtext = ' '.join(filtered_sentence)
        df.loc[index,'alltext'] = newtext
    
    return df

# In[12]:

df = pd.DataFrame(columns=['alltext'])   

uploaded_file = st.file_uploader("Upload Files")
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details) 
    rootdir = "/app/decarbonization/Data/Full_Text/Document" 
    with open(os.path.join(rootdir,uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.success("File_saved")
        
#filename = st.text_input('Enter a file root path:')
#rootdirforfilenames = filename
   
# # Read Data 

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                st.write(filepath)
        if filepath.endswith(".PDF")or filepath.endswith(".pdf"):
            PDF_PATH = filepath.replace('\\','\\')
            print (PDF_PATH)
            start_time = time.time()
            contents = Process_pdf(PDF_PATH)
            df = df.append({'alltext': contents}, ignore_index=True)
            print ("Time taken : " + str(time.time() - start_time))
        elif filepath.endswith(".docx") or filepath.endswith(".DOCX") or filepath.endswith(".doc"):
            DOC_PATH = filepath.replace('\\','\\')
            print (DOC_PATH)
            start_time = time.time()
            contents = Process_Doc(DOC_PATH)
            df = df.append({'alltext': contents}, ignore_index=True)
            print ("Time taken : " + str(time.time() - start_time))

#df.head()   

# # Data Cleansing

#df['alltext'] = df['alltext'].apply(lambda x: clean_it(x))
#df = pd.DataFrame(df['alltext'].str.lower(),columns = ['alltext'])
df.head(2)


# In[17]:


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



# # Apply the model on cleansed data

# In[19]:
#nlp = spacy.load('C:\\Users\\Vikram.V.Kumar\\Desktop\\\Work\\Projects\\GLIMP\\code\\model\myNlp_ensemble')

from spacy_streamlit import visualize_textcat
from spacy_streamlit import visualize_tokens
from spacy_streamlit import load_model


entity_list = pd.DataFrame()
spacy_model = st.sidebar.selectbox("Select NLP Model to be Applied", ["Custom NER", "Custom Text Categorizer", "OOTB NER" ])

if spacy_model == "Custom NER":
        nlp = spacy.load('/app/decarbonization/Model/Decarbon_NER_Model_OOTB')
        for text in df['alltext']:
            doc = nlp(text)            
#            visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
            for ent in doc.ents:
#                st.write(ent.text, ent.start_char, ent.end_char, ent.label_)
                if ent.label_ == 'DECARBON':
                    data = [ent.text]
#                    st.write(ent.text, ent.start_char, ent.end_char, ent.label_)
                    entity_list = entity_list.append(data)
#                    st.write(ent.label_, ent.text)
                    
            entity_list.columns = ['keywords']
            entity_list = entity_list.drop_duplicates()
#            entity_list = entity_list[entity_list["keywords"].str.contains("[^a-zA-Z' ]") == False]
#            entity_list = entity_list[entity_list["keywords"].str.contains("['“()]") == False]
            entity_list = entity_list.head(20)
            st.write(entity_list["keywords"])
else:
    st.write("No Model")



   
















