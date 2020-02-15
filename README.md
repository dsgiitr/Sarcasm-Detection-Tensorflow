#Sarcasm Detection
Source code for Sarcasm Detection  using LSTM and BERT in tensorflow.

News Headlines Dataset has been used for the purpose of Sarcasm Detection. NLP techniques such as LSTM(Long Short Term Memory) and BERT(Bidirectional Encoder Representations from Transformers) have been used for the purpose.

Each record consists of :

is_sarcastic: 1 if the record is sarcastic otherwise 0

headline: the headline of the news article

article_link: link to the original news article. Useful in collecting supplementary data

Pre-trained BERT and LSTM models in tensorflow have been used. The model takes in the headline as the input and outputs 1 if the record is sarcastic, otherwise 0.


#Installation

Clone the git repository:
https://github.com/dsgiitr/Sarcasm-Detection-Tensorflow

Python 3 is the minimum requirement. Tensorflow 1.14.0 has been used. The code would be updated to Tensorflow 2.0


#Working with the Model:
The model can take any statement(either from a dataframe or fed by the user) as the input, and after preprocessing(tokenising etc.).Some functions have been defined for preprocessing the data so that it can be fed in the model and. The model outputs whether the statement is sarcastic or non-sarcastic.
