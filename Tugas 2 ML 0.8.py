#!/usr/bin/env python
# coding: utf-8

# In[1]:


file = open('iris.csv').read().split('\n')[1:]
learning_rate = 0.8


# In[2]:


for i in range(0,len(file)):
    file[i] = file[i].split(',')
    
    for j in range(0,len(file[i])):
        try:
            file[i][j] = float(file[i][j])
        except ValueError:
            pass
        
file = file[:150]
print(file)


# In[3]:


import random

theta_1 = [random.random(), random.random(), random.random(), random.random()]
theta_2 = [random.random(), random.random(), random.random(), random.random()]
bias_1 = random.random()
bias_2 = random.random()

print(theta_1)
print(theta_2)


# In[4]:


category = {
    'setosa': [0,0],
    'versicolor': [0,1], 
    'virginica': [1,1],
}


# In[7]:


import math

avg_error_1 = []
avg_error_2 = []
number_of_correct_predictions = []


for i in range(0,100):
    total_errors = [0,0]
    number_of_correct_prediction = 0
    
    for row in file:
        target_1 = 0
        target_2 = 0

        for i in range(0,4):
            target_1 = target_1+(row[i]*theta_1[i])
            target_2 = target_2+(row[i]*theta_2[i])

        target_1 = target_1 + bias_1
        target_2 = target_2 + bias_2

        sigmoid_1 = 1/(1+math.exp(0-target_1))
        sigmoid_2 = 1/(1+math.exp(0-target_2))

        prediction_1 = round(sigmoid_1)
        prediction_2 = round(sigmoid_2)
        
        if (category[row[4]][0] == prediction_1) and (category[row[4]][1] == prediction_2):
            number_of_correct_prediction = number_of_correct_prediction+1

        error_1 = (sigmoid_1 - category[row[4]][0])**2
        error_2 = (sigmoid_2 - category[row[4]][1])**2
        
        total_errors = [total_errors[0]+error_1,total_errors[1]+error_2]
        
        
        category_1 = category[row[4]][0]
        category_2 = category[row[4]][1]

        d_theta_1 = 2*(sigmoid_1 - category_1)*(1 - category_1)*(sigmoid_1)*row[0]
        d_theta_2 = 2*(sigmoid_1 - category_1)*(1 - category_1)*(sigmoid_1)*row[1] 
        d_theta_3 = 2*(sigmoid_1 - category_1)*(1 - category_1)*(sigmoid_1)*row[2]
        d_theta_4 = 2*(sigmoid_1 - category_1)*(1 - category_1)*(sigmoid_1)*row[3]
        d_theta_5 = 2*(sigmoid_2 - category_2)*(1 - category_2)*(sigmoid_2)*row[0]
        d_theta_6 = 2*(sigmoid_2 - category_2)*(1 - category_2)*(sigmoid_2)*row[1]
        d_theta_7 = 2*(sigmoid_2 - category_2)*(1 - category_2)*(sigmoid_2)*row[2]
        d_theta_8 = 2*(sigmoid_2 - category_2)*(1 - category_2)*(sigmoid_2)*row[3]

        d_bias_1 = 2*(sigmoid_1 - category_1)*(1 - category_1)*(sigmoid_1)
        d_bias_2 = 2*(sigmoid_2 - category_2)*(1 - category_2)*(sigmoid_2)

#         print(theta_1)
#         print(theta_2)

        theta_1 = [
            (theta_1[0] - learning_rate * d_theta_1),
            (theta_1[1] - learning_rate * d_theta_2), 
            (theta_1[2] - learning_rate * d_theta_3), 
            (theta_1[3] - learning_rate * d_theta_4)
        ]

        theta_2 = [
            (theta_2[0] - learning_rate * d_theta_5),
            (theta_2[1] - learning_rate * d_theta_6), 
            (theta_2[2] - learning_rate * d_theta_7), 
            (theta_2[3] - learning_rate * d_theta_8)
        ]

#         print(theta_1)
#         print(theta_2)
    avg_error_1.append(total_errors[0]/150)
    avg_error_2.append(total_errors[1]/150)
    number_of_correct_predictions.append(number_of_correct_prediction)
    print(str([error_1, error_2]))
    
print('\n\n\n\n')
print('Average Error:')
print(avg_error_1)


# In[8]:


import matplotlib.pyplot as plt
plt.plot(range(1,101), avg_error_1)
plt.ylabel('Error_1')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(1,101), avg_error_2)
plt.ylabel('Error_2')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(1,101), number_of_correct_predictions)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# In[ ]:




