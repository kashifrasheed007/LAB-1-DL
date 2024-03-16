#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
scalar1 = tf.constant(5)
scalar2 = tf.constant(6)
mul = scalar1 * scalar2
print(mul)
ssn = tf.compat.v1.Session()
ssn.run(mul)


# In[20]:


vec1 = [1,2,3]
vec2 = [4,5,6]
dot_product = tf.tensordot(vec1, vec2, axes = 1)
ssn = tf.compat.v1.Session()
result = ssn.run(dot_product)
print("Dot Product:", result)


# In[36]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
matrix1 = tf.constant([[1, 2],
                       [3, 4]])
matrix2 = tf.constant([[5, 6],
                       [7, 8]])
add = tf.add(matrix1, matrix2)
sub = tf.subtract(matrix1, matrix2)
mul = tf.multiply(matrix1, matrix2)
div = tf.divide(matrix1, matrix2)

with tf.compat.v1.Session() as sess:
    print("Addition:")
    print(sess.run(add))
    print("Subtraction:") 
    print(sess.run(sub))
    print("Multiplication:")
    print(sess.run(mul))
    print("Division:")
    print(sess.run(div))


# In[38]:


import tensorflow as tf                  # importing tensorflow
vector = tf.constant([1, 2, 3, 4, 5])    # creating a vector
scalar = tf.constant(10)                 # creating a scalar
matrix = tf.constant([[1, 2, 3],         # creating a matrix
                      [4, 5, 6],
                      [7, 8, 9]])
tensor = tf.constant([[[1, 2], [3, 4]],   # creating a tensor
                      [[5, 6], [7, 8]]])
print("Vector:")            #The code imports TensorFlow and creates a vector, scalar, matrix, and tensor      
print(vector)              #using TensorFlow constants.Then, it prints these created objects: vector,                          
print("Scalar:")           #scalar, matrix, and tensor.
print(scalar)
print("Matrix:")
print(matrix)
print("Tensor:")
print(tensor)


# In[48]:


import tensorflow as tf
tensor = tf.constant([[[10, 20], [30, 40]],    # creating a tensor
                      [[50, 60], [70, 80]]])
# usnig the built-in methods of shape, rank, size  
tensor_shape = tf.shape(tensor)
tensor_rank = tf.rank(tensor)
tensor_size = tf.size(tensor)
# using method Start a TensorFlow session
with tf.compat.v1.Session() as sess: 
    # Run the session to evaluate the shape, rank, and size
    shape, rank, size = sess.run([tensor_shape, tensor_rank, tensor_size])
    print("Tensor:")
    print("Shape:", shape)
    print("Rank:", rank)
    print("Size:", size)
    
    #The code utilizes TensorFlow to create 
    #a tensor and then employs built-in methods 
    #to determine its shape, rank, and size. 
    #Finally,it runs a TensorFlow session 
    #to evaluate these properties and prints 
    #the results.


# In[72]:


import tensorflow as tf
shape = [5, 300]   # Define the shape of the tensors
# Create two tensors containing random values between 0 and 1
tensor1 = tf.random.uniform(shape, minval=0, maxval=1)
tensor2 = tf.random.uniform(shape, minval=0, maxval=1)
print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)
#This code segment utilizes TensorFlow to generate two tensors 
#with random values between 0 and 1, each with a shape of [5, 300].
#It then prints both tensors to display their contents.


# In[60]:


import tensorflow as tf
# 1 Create two tensors with random values between 0 and 1 and shape [5, 300]
shape = [5, 300]
tensor1 = tf.random.uniform(shape, minval=0, maxval=1)
tensor2 = tf.random.uniform(shape, minval=0, maxval=1)
# 2 Perform dot product
result = tf.matmul(tensor1, tf.transpose(tensor2))
with tf.compat.v1.Session() as sess:
     # 3 Run the session to evaluate the dot product
    result_value = sess.run(result)
    print("Matrix Multiplication Result:")
    print(result_value)
    
#This code snippet first creates two tensors with random
#values between 0 and 1 and shape [5, 300]. Then, it performs
#a dot product operation between the two tensors using
#`tf.matmul` and `tf.transpose`. Finally, it runs a TensorFlow
#session to evaluate the dot product and prints the result.


# In[63]:


import tensorflow as tf
shape = [5, 300]
tensor1 = tf.random.uniform(shape, minval=0, maxval=1)
tensor2 = tf.random.uniform(shape, minval=0, maxval=1)
tensor2_transposed = tf.transpose(tensor2)
# 2 Perform matrix multiplication
dot_product = tf.tensordot(tensor1, tensor2_transposed, axes=1)
with tf.compat.v1.Session() as sess:
    dot_product_value = sess.run(dot_product)
    print("Dot Product Result:")
    print(dot_product_value)
#The code generates two tensors with random values of shape [5, 300],
#transposes the second tensor, and computes their dot product using 
#`tf.tensordot` with axes set to 1. The resulting dot product is then printed.


# In[64]:


import tensorflow as tf
shape = [224, 224, 3]  #1  Define the shape of the tensor
# 2 Create a tensor containing random values between 0 and 1
random_tensor = tf.random.uniform(shape, minval=0, maxval=1)
print("Random Tensor:")
print(random_tensor)
#This code snippet creates a tensor with a shape of [224, 224, 3]
#containing random values between 0 and 1 using TensorFlow's
#`tf.random.uniform` function. The resulting random tensor is then printed.


# In[65]:


import tensorflow as tf
# Define the shape of the tensor
shape = [224, 224, 3]
# Create a tensor containing random values between 0 and 1
random_tensor = tf.random.uniform(shape, minval=0, maxval=1)
# Calculate the minimum and maximum values
min_value = tf.reduce_min(random_tensor)
max_value = tf.reduce_max(random_tensor)
with tf.compat.v1.Session() as sess:
    # Run the session to evaluate the min and max values
    min_val, max_val = sess.run([min_value, max_value])
    print("Minimum Value:", min_val)
    print("Maximum Value:", max_val)
#This code segment generates a tensor with a shape of [224, 224, 3],
#containing random values between 0 and 1 using TensorFlow's `tf.random.uniform` 
#function. It then calculates the minimum and maximum values within the tensor using
#`tf.reduce_min` and `tf.reduce_max`, respectively. Finally, it runs a TensorFlow 
#session to evaluate and print the minimum and maximum values.


# In[66]:


import tensorflow as tf
shape = [1, 224, 224, 3]
random_tensor = tf.random.uniform(shape, minval=0, maxval=1)
# Squeeze the tensor to change the shape to [224, 224, 3]
squeezed_tensor = tf.squeeze(random_tensor)
with tf.compat.v1.Session() as sess:
    # Run the session to evaluate the squeezed tensor
    squeezed_tensor_value = sess.run(squeezed_tensor)
    print("Shape of Squeezed Tensor:", squeezed_tensor_value.shape)
    print("Squeezed Tensor:")
    print(squeezed_tensor_value)
#This code generates a tensor with a shape of [1, 224, 224, 3] 
#containing random values between 0 and 1 using `tf.random.uniform`. 
#It then squeezes the tensor to change its shape to [224, 224, 3]
#using `tf.squeeze`. Finally, it runs a TensorFlow session to evaluate 
#and print the shape and values of the squeezed tensor.


# In[71]:


import tensorflow as tf
#values for the tensor
values = [5, 8, 12, 7, 3, 9, 15, 6, 11, 10]
# Create a tensor with shape [10] using the defined values
tensor = tf.constant(values)
# Find the index of the maximum value in the tensor
max_index = tf.argmax(tensor)
max_value = tf.reduce_max(tensor)
# Start a TensorFlow session
with tf.compat.v1.Session() as sess:
    # Run the session to evaluate the index of the maximum value
    max_index_value, max_value_value = sess.run([max_index, max_value])
    print("Index of Maximum Value:", max_index_value)
    print("Maximum Value:", max_value_value)
#This code creates a tensor with shape [10] using the defined values
#and finds the index of the maximum value in the tensor using `tf.argmax`.
#Additionally, it calculates the maximum value using `tf.reduce_max`.
#Finally, it runs a TensorFlow session to evaluate and print the index 
#of the maximum value and the maximum value itself.


# In[ ]:





# In[ ]:




