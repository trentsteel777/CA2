import unidecode
import pandas as pd

import glob
import re
import torch

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize

from autocorrect import Speller
from bs4 import BeautifulSoup
from transformers import BertForSequenceClassification, BertTokenizer


from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

def main(folder_path):
    """Uses all the below functions to compute sentiment and the sentiment score(confidence). 
       
       This is the entry point for the application.
    """
    df = text_to_csv(folder_path)
    df['cleaned_content'] = df['content'].apply(text_preprocessing)
    df[['sentiment_label','sentiment_score']] = df['content'].apply(add_sentiment_and_confidence_score)
    return df

def text_to_csv(folder_path):
    """This method reads in all the news articles from the data folder.
    """
    df = pd.DataFrame(columns=['location', 'content'])
    row = 0
    # get all txt files which contain the article information
    for path in glob.glob(folder_path + '/*.txt'):
        with open(path, encoding='utf-8') as f:
            data = f.read()
        dic = {'location': path, 'content': data}
        df = df.append(dic, ignore_index=True)
    return df
    
def text_preprocessing(text, accented_chars=True, lemmatization = True,
                        extra_whitespace=True, newlines_tabs=True, repetition=True, 
                       lowercase=True, punctuations=True, mis_spell=True,
                       remove_html=True, links=True,  special_chars=True,
                       stop_words=True):
  
    """
    This function will preprocess the input text and return the cleaned version of the text.
    """
    stoplist = stopwords.words('english') 
    stoplist = set(stoplist)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    spell = Speller(lang='en')
    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    if newlines_tabs == True: #remove newlines & tabs.
        text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    
    if remove_html == True: #remove html tags
        # Initiating BeautifulSoup object soup.
        soup = BeautifulSoup(text, "html.parser")
        # Get all the text other than html tags.
        text = soup.get_text(separator=" ")
    
    if links == True: #remove links
      # Remove HTML tags using RegEx
      # Removing all the occurrences of links that starts with https
      remove_https = re.sub(r'http\S+', '', text)
      # Remove all the occurrences of text that ends with .com
      text = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    
    if extra_whitespace == True: #remove extra whitespaces
        pattern = re.compile(r'\s+') 
        Without_whitespace = re.sub(pattern, ' ', text)
        # There are some instances where there is no space after '?' & ')', 
        # So I am replacing these with one space so that it will not consider two words as one token.
        text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')

    if accented_chars == True: #remove accented characters
        # Remove accented characters from text using unidecode.
        # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
        text = unidecode.unidecode(text)
    
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()
    
    if repetition == True: #Reduce repetitions   
        # Pattern matching for all case alphabets
        Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

        # Limiting all the  repeatation to two characters.
        text = Pattern_alpha.sub(r"\1\1", text) 

        # Pattern matching for all the punctuations that can occur
        Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')

        # Limiting punctuations in previously formatted string to only one.
        text = Pattern_Punct.sub(r'\1', text)

        # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
        text = re.sub(' {2,}',' ', text)
    
    if punctuations == True: #remove punctuations
        text = re.sub(r"[^a-zA-Z0-9:$-,%.?!]+", ' ', text) 
        
    if stop_words == True: #Remove stopwords
        # The code for removing stopwords
        stoplist = stopwords.words('english') 
        stoplist = set(stoplist)
        text = repr(text)
        # Text without stopwords
        text = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
        # Convert list of tokens_without_stopwords to String type.
        text = ' '.join(text)  
    
    if mis_spell == True: #Check for mis-spelled words & correct them.
        text = spell(text)
    
    if lemmatization == True: #Converts words to lemma form.
        word_list = nltk.word_tokenize(text)
        # Converting words to their root forms
        text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    return text


# Reference: https://github.com/jamescalam/transformers/blob/main/course/language_classification/04_window_method_in_pytorch.ipynb
# This method was copied from line 10 in the above link
def sentiment_preprocessing_for_long_text(txt):
  tokens = tokenizer.encode_plus(txt, add_special_tokens=False,
                               return_tensors='pt')
  input_id_chunks = tokens['input_ids'][0].split(510)
  mask_chunks = tokens['attention_mask'][0].split(510)
  # define target chunksize
  chunksize = 512

  # split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
  input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
  mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

  # loop through each chunk
  for i in range(len(input_id_chunks)):
      # add CLS and SEP tokens to input IDs
      input_id_chunks[i] = torch.cat([
          torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
      ])
      # add attention tokens to attention mask
      mask_chunks[i] = torch.cat([
          torch.tensor([1]), mask_chunks[i], torch.tensor([1])
      ])
      # get required padding length
      pad_len = chunksize - input_id_chunks[i].shape[0]
      # check if tensor length satisfies required chunk size
      if pad_len > 0:
          # if padding length is more than 0, we must add padding
          input_id_chunks[i] = torch.cat([
              input_id_chunks[i], torch.Tensor([0] * pad_len)
          ])
          mask_chunks[i] = torch.cat([
              mask_chunks[i], torch.Tensor([0] * pad_len)
          ])
  return input_id_chunks, mask_chunks

# Reference: https://github.com/jamescalam/transformers/blob/main/course/language_classification/04_window_method_in_pytorch.ipynb
# This method was copied from linkes 11 and 12 in the above link
def add_sentiment_and_confidence_score(data):
    """Uses the `sentiment_pipeline` model to compute sentiment and the confidence score. 
    
    Returns the new dataframe by adding columns containing sentiment and sentiment score
    """
    input_id_chunks,mask_chunks = sentiment_preprocessing_for_long_text(data)
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
                 }
    outputs = model(**input_dict)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    probs = probs.mean(dim=0).detach().numpy()
    probs_dict = {0:'Positive',1:'Negative',2:'Neutral'}

    return pd.Series((probs_dict[probs.argmax()], probs.max()))