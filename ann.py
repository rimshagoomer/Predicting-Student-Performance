import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
import csv
import random
import math
import operator
from sklearn.preprocessing import MinMaxScaler
def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
                if testSet[x] == predictions[x]:
                        correct += 1
        return (correct/float(len(testSet))) * 100.0
              
def backPropagation():
    # fix random seed for reproducibility
    seed=7
    numpy.random.seed(seed)

    trainingSet=[]
    testSet=[]

    dataset = numpy.loadtxt("newfile.csv", delimiter=",")

    #normalize the dataset
    scaler=MinMaxScaler(feature_range=(0, 1))     #between 0 and 1
    dataset=scaler.fit_transform(dataset)

    # split into 67% train and 33% test sets
    train_size=int(len(dataset)*0.67)
    test_size=len(dataset) - train_size
    train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
    trainX=train[:,0:32]
    trainY=train[:,32]
    testX=test[:,0:32]
    testY=test[:,32]   
    print('Creating Neural Network...')
    # create model
    model=Sequential()
    model.add(Dense(55, input_dim=32, init='uniform', activation='sigmoid'))
    model.add(Dense(50, init='uniform', activation='sigmoid'))
    model.add(Dense(35, init='uniform', activation='sigmoid'))
    model.add(Dense(20, init='uniform', activation='sigmoid'))
    model.add(Dense(14, init='uniform', activation='sigmoid'))
    model.add(Dense(8, init='uniform', activation='sigmoid'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    print('Training the model...')
     # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # Fit the model
    model.fit(trainX, trainY, nb_epoch=300, batch_size=15,verbose=0)

    # calculate predictions
    predictions = model.predict(testX)
    # round predictions
    rounded = [round(x[0]) for x in predictions]

    for i in range(len(testY)):
        print(str(i)+'  predicted='+str(rounded[i])+', actual='+ str(testY[i]))
    testScore=getAccuracy(testY,rounded)
    print("Accuracy: %.6f%%" % (testScore))

backPropagation()
