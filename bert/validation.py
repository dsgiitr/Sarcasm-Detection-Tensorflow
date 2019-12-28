"""Validating the model on a sample.
The output should be Sarcastic or Non-Sarcasm
based on the nature of the statement.
(Sarcastic or Non-Sarcastic)."""

examples = convert_text_to_examples(train_text[0], train_labels[0])

(input_ids, input_masks, segment_ids, labels
) = convert_examples_to_features(tokenizer, examples, max_seq_length=max_seq_length)

sentiment = model.predict([input_ids, input_masks, segment_ids])

for i in range(len(sentiment)):
  if(np.argmax(sentiment[i]) == 0):
    print("Non-sarcastic")
  elif (np.argmax(sentiment[i]) == 1):
    print("Sarcasm")
