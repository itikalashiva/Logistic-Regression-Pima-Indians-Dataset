
# Logistic Regression on Pima Indians Diabetes Database dataset

**Abstract**

This project is an effort to use Logistic Regression, written from the ground up, to learn from the Pima

Indians Diabetes Database dataset and predict if a patient has Diabetes or not based on the diagnostic

measures supplied in the dataset. Gradient Descent is used to optimize weights and bias. On the test set,

an accuracy of 80% is obtained.
<hr>
**1.Introduction**

Pima Indians with type 2 diabetes are **metabolically characterized by obesity, insulin resistance**,

insulin secretory dysfunction, and increased rates of endogenous glucose production, which are the

clinical characteristics that define this disease across most populations. In this study, variables such as

glucose levels, skin thickness, and insulin levels are used to determine whether a person has diabetes.

<hr>
**2.Dataset**

Pima Indians Diabetes Database dataset will be used for training, and testing. The dataset contains

medical data of female patients above the age of 21 and 768 instances with the diagnostic measurements

of 8 features. The 8 features are as follows:

1 Glucose (Blood Glucose level)

2 Pregnancies (The number of pregnancies the patient has had)

3 Blood Pressure (mm Hg)

4 Skin Thickness (Triceps skin fold thickness (mm))

5 Insulin level

6 BMI (Body Mass Index: weight in kg/(height in m)2)

7 Diabetes Pedigree Function

8 Age (In years)

<hr>

**3.Data Preprocessing**

The given dataset contains 768 instances, splitting this data into training, test, and validation sets.

Testing data = 60% of Total data

Validation data = 20% of Total data

Test data = 20% of Total data


**3.1 Imputation and Correlation**

There are many instances with zero values for some features, which is practically not possible.

To avoid the data leakage and improve model performance these zero values are imputed with mean

of the respective column.

The correlation between these features improves after imputation.


**3.2 Normalization**

![alt text](https://github.com/itikalashiva/Logistic-Regression-Pima-Indians-Dataset/blob/main/screenshots/beforenorm.PNG)

The values of each feature are in different ranges


From the above data it is observed that each feature has different ranges, so normalizing the data and

bringing every value in the range of 0-1 would help the model to process the data accurately.

**y = (x – min(x))/(max(x) – min(x))**

y: cell value of each column

Every cell value will be normalized, and the range of every column will be between 0 to 1

The normalization is done after the data split, the train data min max values are used to normalize

test and validation data.

<hr>

**4.Model Architecture**

**4.1 Logistic Regression**

Logistic regression is one of the most common machine learning algorithms used for binary

classification. It predicts the probability of occurrence of a binary outcome using a logit function.

It is a special case of linear regression as it predicts the probabilities of outcome using log function.

We use the activation function (sigmoid) to convert the outcome into categorical value.


**4.3 Gradient Descent**

Gradient Descent is used in this project to optimize weights and bias. The gradient of the loss function is

computed, and each weight is reduced by the product of the gradient and the learning rate. The learning

rate is assumed to be 0.1.

<hr>

**5. Hyper Parameters Tuning**

In this project batch size and the learning rate are the hyper parameters, trying to find the weights and bias

for different batch size and learning rates would help us understand what the optimal hyper parameters

are.

iterations = [3000,10000,15000]

learning\_rates = [0.1, 0.05, 0.06]

validation and train loss are calculated with different batch sizes and learning rates, it would give 9 such

graphs for the above mentioned combinations.

After examining the training and validation loss graphs for various batch sizes and learning rates, batch

size =10000 and learning rate =0.1 would result in less loss.

![alt text](https://github.com/itikalashiva/Logistic-Regression-Pima-Indians-Dataset/blob/main/screenshots/traintest.PNG)

<hr>

**6.Results**

**6.1 Training Data & Validation Data**

Batch size =10000 and learning rate =0.1. Initial weights are a zero matrix, and the bias is 0.1

**6.2 Testing**

Batch size =10000 and learning rate =0.1. Initial weights are a zero matrix, and the bias is 0.1

The model gave an accuracy of 80% over the test data, considering any probable value over 0.5 as 1.

<hr>

**7. Evaluation Metrics**

Accuracy is measured by calculating the sum of predicted instances that matches the actual

instances(outcomes) and dividing it by number of outcomes.

![alt text](https://github.com/itikalashiva/Logistic-Regression-Pima-Indians-Dataset/blob/main/screenshots/test.PNG)

<hr>

**Conclusion**

![alt text](https://github.com/itikalashiva/Logistic-Regression-Pima-Indians-Dataset/blob/main/screenshots/final.PNG)

We can confidently state that the model is neither overfitting nor underfitting because the

validation loss was lower than the training loss. Furthermore, on unseen data from the testing set,

the model had an accuracy of 80%. As a result, if the model is used in the real world, it can be

confidently stated that it will perform well.
