#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[26]:


df = pd.read_csv('ex3_train.csv')


# In[27]:


def split_train_test(X, test_ratio):
    np.random.seed(1)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices]


# In[28]:


# train_set, test_set = split_train_test(df, 0.2)
y_train = df.iloc[:,400]
x_train = df.iloc[:,0:400]


# ### Define Functions

# In[29]:


def one_hot_encoding(mat):
    mat = mat.as_matrix()
    temp = []
    for val in mat:
        if val not in temp:
            temp.append(val)
    temp.sort()
    
    result = np.zeros(shape=(len(mat),len(temp)))
    for key, val in enumerate(mat):
        result[key][temp.index(val)] = 1
    return result


# In[30]:


y_train_encoded = one_hot_encoding(y_train)


# In[31]:


y_train_encoded = pd.DataFrame(y_train_encoded)


# In[32]:


def sigmoid(z):
    A = 1. / (1 + np.exp(-z))
#     z_temp = z
    return A


# In[33]:


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
#     print "W1, b1: "
#     print W1.shape, b1.shape
    Z1 = W1.dot(X) + b1
#     print "Z1: "
#     print Z1.shape
    A1 = sigmoid(Z1)
#     print "A1: "
#     print A1.shape
#     print "W2: "
#     print W2.shape
    Z2 = W2.dot(A1) + b2
#     print "Z2: "
#     print Z2.shape
    A2 = sigmoid(Z2)
#     print "A2: "
#     print A2.shape

    assert(A2.shape == (W2.shape[0], A1.shape[1]))
    
    cache_variables = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
#     print "A2 from forward prop:"
#     print A2
    return A2, cache_variables


# In[34]:


def backward_propagation(parameters, cache_variables, X, Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']
  
    A1 = cache_variables['A1']
    A2 = cache_variables['A2']
    
    dZ2= np.array(A2 - Y)
    #dZ2 (10,2800)
    #A1 (25,2800)
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), A1 * (1 - A1))
    dZ1 = np.array(dZ1)
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return gradients


# In[35]:


def softmax(x):
    exp_scores = np.array(np.exp(x))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y_predict = np.array(np.argmax(probs, axis=0))
    return y_predict


# ### 2. Split the Data

# - Please refer to cell 3, 4 and 5

# ### 3. Initilize parameters

# In[36]:


def initialize_parameters(n_x, n_h, n_y):
   
    np.random.seed(1) 
    
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
   
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[37]:


def calculate_cost(A2, Y, parameters):
    
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A2 = np.matrix(A2)
    #print A2
#     logprobs = Y.dot(np.log(A2).T) + np.multiply((1 - Y), np.log(1 - A2))
#     cost = - np.sum(logprobs) / m
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    
    cost = np.squeeze(cost) 
    #print type(cost)
    #print cost
    assert(isinstance(cost, float))
    return cost


# ### 4. Neural Network Model with 1 hidden layer

# In[38]:


def update_parameters(parameters, gradients, learning_rate):
    
    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']
   
    parameters["W1"] = parameters["W1"] - learning_rate * dW1
    parameters["b1"] = parameters["b1"] - learning_rate * db1
    parameters["W2"] = parameters["W2"] - learning_rate * dW2
    parameters["b2"] = parameters["b2"] - learning_rate * db2
   
    #parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters


# In[39]:


def layer_sizes(X, Y):
    
    n_x = X.shape[0] 
    n_h = 25
    n_y = Y.shape[0] 
    
    return (n_x, n_h, n_y)


# In[40]:


def neuralnetwork_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    
    np.random.seed(1)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    print n_x, n_y
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    costs = []
    for i in range(0, num_iterations):
        A2, cache_variables = forward_propagation(X, parameters)
        cost = calculate_cost(A2, Y, parameters)
        gradients = backward_propagation(parameters, cache_variables, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)
        if print_cost and i % 1000 == 0:
            print ("Cost after %i iterations: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs


# ### 5. Predictions

# In[41]:


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
#     exp_scores = np.array(np.exp(cache["Z2"]))
#     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y_predict = softmax(cache["Z2"])
    return y_predict


# ### 6. Optimization

# In[18]:


learning_rate = 0.0001
parameters0, costs0 = neuralnetwork_model(x_train.T, y_train_encoded.T, n_h = 25, num_iterations=5000, print_cost=True)
learning_rate = 0.01
parameters1, costs1 = neuralnetwork_model(x_train.T, y_train_encoded.T, n_h = 25, num_iterations=5000, print_cost=True)
learning_rate = 0.001
parameters2, costs2 = neuralnetwork_model(x_train.T, y_train_encoded.T, n_h = 25, num_iterations=5000, print_cost=True)
# plt.plot(np.squeeze(costs0))
# plt.plot(np.squeeze(costs1))
# plt.plot(np.squeeze(costs2))
# plt.ylabel('cost')
# plt.xlabel('iterations (per tens)')
# plt.title("Cost vs Number of Iterations")
# plt.legend(['0.0001', '0.01', '0.001'], loc='upper left')
# plt.show()


# #### Optimizied learning rate and iterations

# In[19]:


learning_rate = 0.5
parameters3, costs3 = neuralnetwork_model(x_train.T, y_train_encoded.T, n_h = 25, num_iterations=10000, print_cost=True)


# In[20]:


# plt.plot(np.squeeze(costs3))
# plt.ylabel('cost')
# plt.xlabel('iterations (per tens)')
# plt.title("Cost vs Number of Iterations")
# plt.show()


# In[21]:


predictions = predict(parameters3, x_train.T)


# In[22]:


#predictions = predict(parameters3, x_train.T)
correct = [1 if a == b else 0 for (a, b) in zip(y_train, predictions)]  
accuracy = (float(sum(map(int, correct))) / float(len(correct)))  
print 'Train set accuracy = {0}%'.format(accuracy * 100)


# #### Test set 

# In[23]:


df1 = pd.read_csv('ex3_test.csv')
x_test = df1.iloc[:,0:400]
y_test = df1.iloc[:,400]


# In[24]:


predictions1 = predict(parameters3, x_test.T)
correct = [1 if a == b else 0 for (a, b) in zip(y_test, predictions1)]  
accuracy = (float(sum(map(int, correct))) / float(len(correct)))  
print 'Optimized test set accuracy = {0}%'.format(accuracy * 100)


# In[ ]:




