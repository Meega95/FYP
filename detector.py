import streamlit as st
import joblib
import re
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity
import emoji
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spell checkers
spell_checker = SpellChecker()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary('en-80k.txt', term_index=0, count_index=1)

def text_cleaning(text):
    text = emoji.demojize(text)                                                 # Decode emojis   
    text = BeautifulSoup(text, 'html.parser').text                              # Remove HTML tag    
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'http\S+', '', text)                                         # Remove URLs   
    text = text.lower()                                                         # Lower the text    
    text = text.translate(str.maketrans('', '', string.punctuation))            # Remove punctuation
    text = re.sub(r'\d+', '', text)                                             # Remove digits   
    words = word_tokenize(text)                                                 # Tokenize the text to words                                                                           
    stop_words = set(stopwords.words('english'))                                # Remove stopwords    
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()                                            # Lemmatize the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Remove 'ing' suffix where appropriate
    def remove_ing_suffix(word):
        if word.endswith('ing') and len(word) > 4:
            return word[:-3]
        return word
    words_no_ing = [remove_ing_suffix(word) for word in lemmatized_words]

    def correct_spelling(word):             # Check and correct spelling
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)        # Use SymSpell
        if suggestions:
            return suggestions[0].term
        candidates = spell_checker.candidates(word)                             # Use SpellChecker
        if candidates:
            # Create a dictionary with candidate words and their scores
            candidate_scores = {candidate: spell_checker.word_frequency[candidate] for candidate in candidates}
        blob = TextBlob(word)                                                   # Use TextBlob
        return str(blob.correct())
    corrected_words = [correct_spelling(word) for word in lemmatized_words]
    # Join the words back into a single string
    processed_text = ' '.join(corrected_words)
    return processed_text

def text_process(text):
    nopunc = [char for char in user_input if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load the model
pipeline_svc_best = joblib.load('pipeline_svc_best_random.pkl')

# Streamlit app
st.title('Offensive Tweet Classifier')
st.write('Enter a tweet and click the button to see if it is offensive or not.')

# Input text box
user_input = st.text_area('Enter tweet:')

# Prediction button
if st.button('Predict'):
    # Preprocess user input
    processed_input = text_cleaning(user_input)

    # Perform prediction
    prediction = pipeline_svc_best.predict([processed_input])[0]
    
    # Determine result based on prediction
    if prediction == 'Offensive':
        result = 'Offensive'
    elif prediction == 'Normal':
        result = 'Normal'
    else:
        result = 'Uncertain'
    
    # Display the prediction result
    st.write(f'The tweet is: **{result}**')

