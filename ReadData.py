import numpy as np
import matplotlib.pyplot as mp


data_file=open("C:/Users\Arno/PycharmProjects/KI/mnist_train_100.csv","r")
data_list=data_file.readlines()
data_file.close()

all_values = data_list[0].split(",")
image_array=np.asfarray(all_values[1::]).reshape((28,28))
scaled_input=(np.asfarray(all_values[1::])/255.0*0.99)+0.1
print(scaled_input)
#mp.imshow(image_array,cmap="Greys",interpolation="None")
