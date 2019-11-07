#Implement Fisher Algroithm to classify between digits from 0 ----> 9 (10 Classes)


import numpy as np
import matplotlib.pylab as plt
import imageio
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Training Phase

#1- Read and load training images 
#2- Flatten  training Images

training_img_path = 'E:\\Train\\'
list_of_images_training = []
for i in range(1,2401): 
    im_path = training_img_path + '{}.jpg'.format(i)
    im = imageio.imread(str(im_path))
    im_numpy = np.array(im)
    im_numpy_flatten = im_numpy.flatten().reshape(1,784)
    list_of_images_training.append(im_numpy_flatten)	


list_of_images_training_np = np.array(list_of_images_training).reshape(2400,784)
#print(list_of_images_training_np.shape)               ### (2400,784)


#Function to be used to seperate classes 

def seperate_class_help (list_of_images_training):
	needed_class_list  = [list_of_images_training[start:start+240] for start in range(0,len(list_of_images_training),240)]
	return needed_class_list


##Mean_fun to calculate mean of a list
    
def mean_fun(list):
    mean_val = np.round((np.sum(list, axis = 0, keepdims = True))/(len(list)),decimals=8)
    return mean_val



#3- generate whole list of images seperate using seperate_class_help function
#4- Loop over each class and seperate versus others
#5- Calculate mean using mean_fun function
#6- Calculate Sw & Sw_inv
#7- Calculates Wieght matrices W1_All & Bias W0_All for all classes

W0_All = []
W1_All = []
CL_a_list=[]
CL_b_list=[]  
M_a_list=[]
M_b_list=[]
S_a = np.zeros((784,784))
S_b = np.zeros((784,784))

data_whole_list = seperate_class_help(list_of_images_training_np)
#Sw = np.zeros((784,784))
for i in range(10):
#    print(i)
    CL_a = data_whole_list[i]
    CL_a_list.append(CL_a)
   
    M_a = mean_fun(CL_a_list[i])
    M_a_list.append(M_a)
    
    CL_b = [value for index,value in enumerate(data_whole_list) if index!= i]
    CL_b_array=np.array(CL_b).reshape(2160,784)
    CL_b_list.append(CL_b_array)
       
    M_b = mean_fun(CL_b_list[i])
    M_b_list.append(M_b)
    
 #   print(M_a.shape)
  
    S_a = S_a + np.dot((np.subtract(CL_a_list[i], M_a_list[i])).T.reshape(784,240), np.subtract(CL_a_list[i], M_a_list[i]).reshape(240,784))
    S_b = S_b + np.dot((np.subtract(CL_b_list[i], M_b_list[i])).T.reshape(784,2160), np.subtract(CL_b_list[i],M_b_list[i]).reshape(2160,784))

    Sw = S_a + S_b
    Sw_inv = np.linalg.pinv(Sw)
    
    # w 1*784
    #w0 scalar
    
    W1 = np.dot(Sw_inv,(np.subtract(M_b_list[i],M_a_list[i]).T))
    W1_All.append(W1)
    W0 = -.5 * np.dot(sum(M_a_list[i], M_b_list[i]),np.array(W1))
    W0_All.append(W0)

#print(np.array(W1_All).shape)
#print(W0_All)
#print(np.array(W0_All).shape)

#8- Load Training Labels    
#9-Calculate training data accuracy

training_label_file_path = 'Training Labels.txt'
training_digits = []
training_labels = np.loadtxt(training_img_path+training_label_file_path)
for v in range(len(training_labels)):
    Y_label_training = []
    for n in range(10):
        Y_training = (np.dot(list_of_images_training[v],W1_All[n])) + W0_All[n]
        Y_label_training.append(Y_training)
    training_digits.append(Y_label_training.index(min(Y_label_training)))

print('Accuracy of Training Phase= ', round((accuracy_score(training_digits,training_labels)*100),0)) 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Testing phase

#1- Load all images that will be used in test
#2- Flatten images


testing_img_path = 'E:\Test\\'
Y_All = []
list_of_images_testing = []
for u in range(1,201): 
    im_path_test = testing_img_path+ '{}.jpg'.format(u)
    im_test = imageio.imread(str(im_path_test))
    im_numpy_test = np.array(im_test)
    im_numpy_flatten_test = im_numpy_test.flatten().reshape(1,784)
    list_of_images_testing.append(im_numpy_flatten_test)
list_of_images_testing_np = np.array(list_of_images_testing).reshape(200,784)

#print(Y_All)
#print(np.array(Y_All).shape)


#3- load all test labels 
#4- Calculate perdicted Y values
#5- Calculate accuracy of testing phase

training_label_file_path = 'Test Labels.txt'
testing_digits = []
testing_labels = np.loadtxt(testing_img_path+training_label_file_path)
for v in range(len(testing_labels)):
#    print(v)
    Y_label_testing = []
    for n in range(10):
#        print(n)
        Y_testing = (np.dot(W1_All[n].T,list_of_images_testing_np[v])) + W0_All[n]
        Y_label_testing.append(Y_testing)
        #Y_label_testing_array=np.array(Y_label_testing))#.reshape(10,1)
    testing_digits.append(np.argmin(Y_label_testing))
    #testing_digits.append(Y_label_testing.index(min(Y_label_testing)))
print('Accuracy of testing Phase= ', round((accuracy_score(testing_digits,testing_labels)*100),0))

#6- generate confusion matrix
#7- plot confusion matrix as a heatmap

test_confusion_matrix = confusion_matrix(testing_labels, testing_digits)
# function to plot confusion matrix as heat map
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

print_confusion_matrix(test_confusion_matrix, [0,1,2,3,4,5,6,7,8,9])
