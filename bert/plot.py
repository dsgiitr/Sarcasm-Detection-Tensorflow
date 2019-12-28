
import matplotlib.pyplot as plt #Library for plotting in python
# summarize history for accuracy
plt.plot(model_fit.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('bert_accuracy.jpeg')



# summarize history for loss
plt.plot(model_fit.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('bert_loss.jpeg')
