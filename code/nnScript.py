import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    s = 1.0 / (1.0 + np.exp(-1.0 * z));
    return s 



def preprocess():
    """ Input: Loading MNIST file

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    selected_features=[]
    features=np.where(~train_data.any(axis=0))[0]
    for i in range(0,784):
        if i not in features:
            selected_features.append(i) 


    train_data=np.delete(train_data,features,axis=1)
    validation_data=np.delete(validation_data,features,axis=1)

    test_data=np.delete(test_data,features,axis=1)

    print(train_data.shape)
    print(len(selected_features))


    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label,selected_features


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
   

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0




    # FORWARD PROPAGATION BEGINS #
    size=training_data.shape[0]
    bias_1=np.ones((size,1))
    #bias_1=np.ones((1,size))
    training_data= np.concatenate((bias_1,training_data),axis=1)
    w1_transpose=np.transpose(w1)
    data_product=np.dot(training_data,w1_transpose)
    Z_sigmoid=sigmoid(data_product)

    #print(Z_sigmoid)
    
    size_of_sigmoid=Z_sigmoid.shape[0]
    bias_2=np.ones((size_of_sigmoid,1))
    #bias_2=np.ones(len(Z_sigmoid))
    Z_sigmoid= np.concatenate((bias_2,Z_sigmoid),axis=1)
    w2_transpose=np.transpose(w2)
    output_temp=np.dot(Z_sigmoid,w2_transpose)
    output=sigmoid(output_temp)
    
    #training_label=output.argmax(axis=1)
    
    
    # BACK PROPAGATION BEGINS #
     
    c=training_label[0]
    yl_matrix=np.zeros((1,n_class))
    yl_matrix[0][c]=1
    size=training_data.shape[0]
    
    l=training_data.shape[0]
    yl_matrix=np.zeros([l,n_class])
    yl_matrix[np.arange(l),train_label.astype(int)]=1

    delta_l=output-yl_matrix
    grad_w2=np.dot(delta_l.transpose(),Z_sigmoid)
    grad_w2=(grad_w2+(lambdaval*w2))/n_input
    
    
    dot_product=np.dot(delta_l,w2)
    Z=(1-Z_sigmoid)*Z_sigmoid
    semi_final=Z*dot_product
    semi_final=np.transpose(semi_final)
    grad_w1=np.dot(semi_final,training_data)
    
    grad_w1=np.delete(grad_w1,0,axis=0)
    
    grad_w1=(grad_w1+(lambdaval*w1))/n_input
    
    obj_grad = (np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0))#/n_input
   
    # ERROR FUNCTION #
    err_funct=(yl_matrix*np.log(output))+((1-yl_matrix)*np.log(1-output))
    err_funct_sum=np.sum(err_funct)
    err=-(err_funct_sum)/n_input
    
    W1_final=np.sum(np.square(w1))
    W2_final=np.sum(np.square(w2))
    WEIGHT=W1_final+W2_final
    obj_val=err+((lambdaval/(2*n_input))*WEIGHT) #REGULARIZATION


    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    size=data.shape[0]

    bias_1=np.ones((size,1))
    data=np.concatenate((bias_1,data),axis=1)
    w1_transpose=np.transpose(w1)
    data_product=np.dot(data,w1_transpose)
    Z_sigmoid=sigmoid(data_product)
    size_of_sigmoid=Z_sigmoid.shape[0]
    bias_2=np.ones((size_of_sigmoid,1))
    
    Z_sigmoid=np.concatenate((bias_2,Z_sigmoid),axis=1)
    w2_transpose=np.transpose(w2)
    sigmoid_product=sigmoid(np.dot(Z_sigmoid,w2_transpose))


    labels=sigmoid_product.argmax(axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label,selected_features = preprocess()

            #  Train Neural Network

            # set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
#for i in np.arange(36,84,4):
n_hidden = 80
    #print(n_hidden)


                    # set the number of nodes in output unit
n_class = 10

                    #initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)


                    # unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

                    #print(initialWeights.shape)

                    # set the regularization hyper-parameter
        #for i in np.arange(0,65,5):
lambdaval = 5
                #print(lambdaval)


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)


opts = {'maxiter': 50}  # Preferred value.
                                        
                                        
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

                                      

                                        # Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

                                        # Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)


                                        # find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

                                            # find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

                                            # find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
obj = [selected_features, n_hidden, w1, w2, lambdaval]
                                            # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
pickle.dump(obj, open('params.pickle', 'a+b'))
