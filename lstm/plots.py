import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model_fit.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
from google.colab import files
plt.savefig('lstm_accuracy.jpeg')
files.download('lstm_accuracy.jpeg')
# summarize history for loss

plt.plot(model_fit.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
from google.colab import files
plt.savefig('lstm_loss.jpeg')
files.download('lstm_loss.jpeg')
