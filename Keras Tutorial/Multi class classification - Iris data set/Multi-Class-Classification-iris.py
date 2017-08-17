# Muliti class classification using Keras, 
# steps 
    # Import useful packages
    # fix seed values 
    # Load the data
    # Categorical values to Dummies (one-hot encoder)
    # Define the Neural Network Model
    # 

## Step 1. Import Usefull packages & classes
import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

## Step 2. Fix the Seed value 
seed = 7
numpy.random.seed(seed)

## Step 3. Load dataset

dataFrame = pandas.read_csv("Iris.csv", header=1)
dataset = dataFrame.values

# Segrigating the X (independent values[ all rows, 4 columns ]) & Y (dependent variable[5th column])
X = dataset[:,1:5].astype(float)
Y = dataset[:,5]
# Check X & Y values
# print(X)
# print(Y)

# Step 4. One-hot encoding for 3 categorical values
    # sertosa, versicolor verginica
    #    1        0         0      -> Sertosa
    #    0        1         0      -> versicolor
    #    0        0         1      -> verginica

encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
# print(encoder_Y) -> encoded values with 3 categories setosa ->1, versicolor -> 2, virginica -> 3  

# convert integers to dummy variables(ie. one hot encoded)
dummy_y = np_utils.to_categorical(encoder_Y)
# print(dummy_y) -> converts into one hot encoding 

## Step 4. Neural network model.
def NeuralNets_model():
    # Define the sequence and add layers to the neural network
    model = Sequential()
    model.add(Dense(8, init='glorot_uniform', input_dim = 4, activation = 'relu' ))
    model.add(Dense(3, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# estimator 
estimator = KerasClassifier(build_fn=NeuralNets_model,nb_epoch=150, batch_size =10)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("NeuralNets_model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))