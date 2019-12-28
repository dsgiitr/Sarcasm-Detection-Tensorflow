"""Validating the model on a sample.
The output would be either 'Sarcasm' or
'Non-Sarcastic' depending on the nature
of Statement(sarcastic or non-sarcastic). """


headline = df['headline'][0]
headline = tokenizer.texts_to_sequences(headline)
headline = pad_sequences(headline, maxlen=78, dtype='int32', value=0)

sentiment = model.predict(headline,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("Non-sarcastic")
elif (np.argmax(sentiment) == 1):
    print("Sarcasm")
