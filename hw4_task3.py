import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
# ......
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.2
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.0001, 0.01, 1, 100, 10000]
# --- end of task --- #

er_valid_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
        # Split the training rows into 5 folds.
    k = 5
    folds = np.array_split(np.arange(num_train), k)
    fold_errors = []

    for i in range(k):
        # Use one fold for validation and the others for training.
        valid_indices = folds[i]
        train_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        model.fit(
            sample_train[train_indices],
            label_train[train_indices]
        )

        predictions = model.predict(sample_train[valid_indices])

        # Calculate mean squared error for this validation fold.
        fold_error = np.mean(
            (predictions - label_train[valid_indices]) ** 2
        )
        fold_errors.append(fold_error)

    er_valid = np.mean(fold_errors)
    er_valid_alpha.append(er_valid)

    print("Alpha:", alpha, "Validation error:", er_valid)
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)]
print("Best alpha:", alpha_opt)

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
model.fit(sample_train, label_train)
pred_train = model.predict(sample_train)
pred_test = model.predict(sample_test)
er_train = np.mean((pred_train - label_train) ** 2)
er_test = np.mean((pred_test - label_test) ** 2)

print("Training error:", er_train)
print("Testing error:", er_test)
