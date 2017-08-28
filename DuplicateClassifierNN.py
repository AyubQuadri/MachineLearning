import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical


def splitData(data,testSize):
  np.random.seed(10)
  trainData,testData = train_test_split(data, test_size = testSize)
  return trainData,testData

def buildNeuralnet(x_train, y_train,x_test, y_test):

    model = Sequential()
    model.add(Dense(27, input_dim=27, init="uniform", activation='relu'))
    model.add(Dense(1,  activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
   
    model.fit(x_train, y_train,
              batch_size=128,
              nb_epoch =1,
              verbose=0,
              validation_data= (x_test,y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


milkData = pd.read_csv("quora_features_all.csv",sep=",")


milkData = milkData.drop(['question1'],axis=1).drop(['question2'],axis=1)

train,test = splitData(milkData,0.2)
#
trainLabels = train['is_duplicate']
trainData = train.drop(['is_duplicate'],axis=1)

testLabels = test['is_duplicate']
testData = test.drop(['is_duplicate'],axis=1)

trainData = np.array(trainData)
trainLabels = np.array(trainLabels)

testData = np.array(testData)
testLabels = np.array(testLabels)
buildNeuralnet(trainData,trainLabels,testData,testLabels)