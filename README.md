# Diplomarbeit

Die Basis des Codes die verwendet wurde für die Diplomarbeit:
https://www.kaggle.com/code/stpeteishii/b200c-lego-classify-conv2d

## Reschersche
https://www.blackbox.ai/
https://www.kaggle.com

## Entwicklung & Implementierung
```
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
data_dir = '../input/b200c-lego-classification-dataset/64'
Name=[]
for file in os.listdir(data_dir):
    Name+=[file]
print(Name)
print(len(Name))
200
N=[]
for i in range(len(Name)):
    N+=[i]
    
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 
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
n=len(datax0)
M=[]
for i in range(n):
    M+=[i]
random.shuffle(M)
datax1=np.array(datax0)
datay1=np.array(datay0)
trainx0=datax1[M[0:(n//4)*3]]
testx0=datax1[M[(n//4)*3:]]
trainy0=datay1[M[0:(n//4)*3]]
testy0=datay1[M[(n//4)*3:]]
trainy2=to_categorical(trainy0)
X_train=np.array(trainx0).reshape(-1,10,10,3)
y_train=np.array(trainy2)
X_test=np.array(testx0).reshape(-1,10,10,3)
trainx,testx,trainy,testy=train_test_split(X_train,y_train,test_size=0.2,random_state=44)
print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)
(480000, 10, 10, 3)
(120000, 10, 10, 3)
(480000, 200)
(120000, 200)
model = Sequential()
model.add(Conv2D(32,(4,4),input_shape = (10,10,3),activation = 'relu'))
model.add(Conv2D(32,(2,2),activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
Model: "sequential"

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



