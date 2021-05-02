#!/usr/bin/env python
# coding: utf-8

# ## JAYDEEP CHAUDHARI-Task 1: PREDICTION USING SUPERVISED ML
# 
# ### TASK OBJECTIVE: PREDICT THE PERCENTAGE OF STUDENTS BASE ON NUMBER OF HOURS OF STUDY AND PREDICT THE SCORE IF A STUDENT STUDIES FOR 9.25 HRS/DAY 

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)



# In[5]:


s_data.head(10)


# ### Visualization of data
# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[6]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# ### **Preparing the data**
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[7]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# ### Splitting data into train test model

# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[8]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[9]:


X_train


# In[10]:


X_train


# ### **Training the Algorithm**
# ### Here I am using linear regression model as we can see direct relation between inputs and outputs
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[14]:


from sklearn.linear_model import LinearRegression  
lf = LinearRegression()  
lf.fit(X_train, y_train) 


# In[15]:


# Plotting the regression line
line = lf.coef_*X+lf.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[16]:


print(X_test) # Testing data - In Hours
y_pred = lf.predict(X_test) # Predicting the scores


# In[17]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[19]:


# You can also test with your own data
hours = 9.25
own_pred = lf.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### **Evaluating the model**
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




