import altair as alt
import pandas as pd
import streamlit as st
import os
import pickle
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
from tensorflow.keras.models import load_model
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_tf_bert import TFBertModel
nltk.data.path.append('/usr/share/nltk_data')



# Get the directory of the current script (src directory)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the savedfiles directory (sibling to this script)
savedfiles_path = os.path.join(current_dir, "savedfiles")

ps = PorterStemmer()
def get_sentense_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y.copy()
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y.copy()
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

@st.cache_resource
def get_model():
    return load_model(
        savedfiles_path+"/bert_dense_model.h5",
        custom_objects={"TFBertMainLayer": TFBertModel}
    )

def bert_pred(msg):
    # Load model
    model = get_model()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(savedfiles_path+"/bert_tokenizer/")
    # Tokenize
    inputs = tokenizer(
        [msg],
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    # Predict
    inp_dict = {
        'input_word_ids': inputs['input_ids'],
        "input_mask": inputs['attention_mask'],
        "input_type_ids": inputs['token_type_ids']
    }
    y_pred = model.predict(inp_dict)
    result = np.argmax(y_pred, axis=1)[0]
    return result

@st.cache_resource
def get_w2v_model():
    return Word2Vec.load(current_dir+"/w2v_model.model")
@st.cache_resource
def get_rf():
    return pickle.load(open(current_dir+'/rf_model.pkl', 'rb'))

def word_2_vec(msg):
    w2v_model = get_w2v_model()
    model = get_rf()

    # 1. preprocess
    input_message = transform_text(msg)

    # 2. vectorize
    input_vector = get_sentense_vector(input_message.split(), w2v_model).reshape(1, -1)

    # 3. predict
    result = model.predict(input_vector)[0]
    return result

# display
def main():
    st.title("Message Spam Classifier")
    input_message = st.text_input("Enter the message")
    embeds = st.selectbox('Select Embedding', ['Word2Vec', 'Bert'])
    if st.button('Check'):
        st.write('Please wait while model is predicting..')

        if embeds == 'Bert':
            result = bert_pred(input_message)
        else:
            result = word_2_vec(input_message)
        
        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
        
        
if __name__ == "__main__":
    main()