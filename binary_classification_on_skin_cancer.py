import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import Xception

ben = os.listdir('../input/skin-cancer-malignant-vs-benign/train/benign')
mal = os.listdir('../input/skin-cancer-malignant-vs-benign/train/malignant')

# Let benign be 0 and malignant be 1
train = []
train_y = []

# loading the dataset
for i in ben:
    x = '../input/skin-cancer-malignant-vs-benign/train/benign/' + i
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(200,200))
    img = img/255 # normalising
    train.append(img.flatten())
    train_y.append(0)

for i in mal:
    x = '../input/skin-cancer-malignant-vs-benign/train/malignant/' + i
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(200,200))
    img = img/255 # normalising
    train.append(img.flatten())
    train_y.append(1)

train = np.array(train)

# Splitting the dataset
train,val,train_y,val_y = train_test_split(train,train_y,test_size=0.2,random_state=44)
train = train.reshape(train.shape[0],200,200,3)
val = val.reshape(val.shape[0],200,200,3)
encoder = LabelEncoder()
encoder = encoder.fit(train_y)
train_y = encoder.transform(train_y)
encoder = encoder.fit(val_y)
val_y = encoder.transform(val_y)
print(str('training rows ' + str(len(train))))
print(str('validation rows ' + str(len(val))))

# neural network model creation
model = Sequential()
base = Xception(include_top=False,weights="imagenet",input_shape=(200,200,3),pooling='avg')
for lay in base.layers: lay.trainable = True
model.add(base)
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(64,'relu'))
model.add(Dropout(0.6))
model.add(Dense(1,'sigmoid'))
model.compile(Adam(0.0001),'binary_crossentropy',['accuracy'])
model.summary()

# defining callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=2,verbose=2,factor=0.3,min_lr=0.000001)
early_stop = EarlyStopping(patience=4,restore_best_weights=True)

# training the model
model.fit(train,train_y,epochs=25,batch_size=10,validation_data=(val,val_y),verbose=2,callbacks=[early_stop,reduce_lr])

# testing the trained model
ben = os.listdir('../input/skin-cancer-malignant-vs-benign/test/benign')
mal = os.listdir('../input/skin-cancer-malignant-vs-benign/test/malignant')
test = []
test_y = []

for i in ben:
    x = '../input/skin-cancer-malignant-vs-benign/test/benign/' + i
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(200,200))
    img = img/255 # normalising
    test.append(img)
    test_y.append(0)
for i in mal:
    x = '../input/skin-cancer-malignant-vs-benign/test/malignant/' + i
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(200,200))
    img = img/255 # normalising
    test.append(img)
    test_y.append(1)

test = np.array(test)
encoder = LabelEncoder()
encoder = encoder.fit(test_y)
test_y = encoder.transform(test_y)

loss,acc = model.evaluate(test, test_y,verbose=2)

print('Accuracy on test data: '+ str(acc*100))
print('Loss on test data: ' + str(loss))