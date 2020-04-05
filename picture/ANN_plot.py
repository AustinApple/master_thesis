import keras 
from keras.models import Sequential 
from keras.layers import Dense 
from ann_visualizer.visualize import ann_viz 

network = Sequential() 
        #Hidden Layer#1
network.add(Dense(units=4,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=3)) 

        #Hidden Layer#2
network.add(Dense(units=4,
                  activation='relu',
                  kernel_initializer='uniform')) 

        #Exit Layer
network.add(Dense(units=2,
                  activation='sigmoid',
                  kernel_initializer='uniform')) 



ann_viz(network, filename="test.gv")