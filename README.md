# Diplomarbeit

Die Basis des Codes die verwendet wurde für die Diplomarbeit:
https://www.kaggle.com/code/stpeteishii/b200c-lego-classify-conv2d

## Reschersche
https://www.blackbox.ai/
https://www.kaggle.com

## Entwicklung & Implementierung

### Abgeänderter Code

### Wichtigste stellen des Original-Codes
```
import numpy as np...

data_dir = '../input/b200c-lego-classification-dataset/64'

datax0=[]
datay0=[]
count=0
for file in tqdm(Name):
    path=os.path.join(data_dir,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(10,10))
        image=img_to_array(image)
        image=image/255.0
        datax0.append(image)
        datay0.append(count)
    count=count+1
100%|██████████| 200/200 [1:09:53<00:00, 20.97s/it]


model = Sequential()
model.add(Conv2D(32,(4,4),input_shape = (10,10,3),activation = 'relu'))
model.add(Conv2D(32,(2,2),activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


his = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=64, verbose=2)

y_pred=model.predict(testx)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(testy,axis=1)
print(classification_report(ground,pred))
            

get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

<Figure size 432x288 with 0 Axes>
epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

<Figure size 432x288 with 0 Axes>
pred2=model.predict(X_test)
print(pred2[0:10])

PRED=[]
for item in pred2:
    value2=np.argmax(item)      
    PRED+=[value2]
print(PRED[0:10])

  2.6869937e-15 4.3437779e-12]]
[143, 179, 159, 75, 30, 7, 79, 35, 88, 116]
ANS=list(testy0)
ANS[0:10]
[182, 91, 144, 109, 30, 7, 44, 35, 165, 116]
accuracy=accuracy_score(ANS,PRED)
print(accuracy)
0.46362
 
 
```



