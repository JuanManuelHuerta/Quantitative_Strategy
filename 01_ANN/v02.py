import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from SYF_ANN import *


normalization=16.0
    
# STEP 1: Load data, produce one-hot encoding of targets and split into training and testing
dig = load_digits()
onehot_target = [[1 if y==x else 0 for y in range(10)] for x in dig.target]
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

## STEP 2: Create network, fit the data
model = SYF_ANN(x_train/normalization, np.array(y_train))
model.fit(epochs=1500)

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

## STEP 3: Bencmkark Accuracy
print("Training accuracy : ", get_acc(x_train/normalization, np.array(y_train)))
print("Test accuracy : ", get_acc(x_val/normalization, np.array(y_val)))
