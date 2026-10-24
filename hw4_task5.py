import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# ......
# --- end of task --- #

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# vary the percentage of data for training
num_train_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per: 

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression(max_iter=5000)

    # --- Your Task --- #
        # Train the baseline model.
    model.fit(sample_train, label_train)

    # Calculate testing accuracy.
    pred_base = model.predict(sample_test)
    acc_base = np.mean(pred_base == label_test)
    acc_base_per.append(acc_base)

    # Calculate testing AUC using predicted probabilities.
    prob_base = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, prob_base)
    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    # --- Your Task --- #
    # Repeat minority-class training rows to balance the classes.
    positive = np.where(label_train == 1)[0]
    negative = np.where(label_train == 0)[0]

    rng = np.random.default_rng(42)
    extra_positive = rng.choice(
        positive,
        size=len(negative) - len(positive),
        replace=True
    )

    indices = np.concatenate([
        np.arange(num_train), extra_positive
    ])

    model_yours = LogisticRegression(max_iter=5000)
    model_yours.fit(sample_train[indices], label_train[indices])

    # Get positive-class probabilities
    prob_yours = model_yours.predict_proba(sample_test)[:, 1]

    # Correct probabilities for the original training class balance
    prior = np.mean(label_train)
    prob_corrected = (
        prob_yours * prior
        / (prob_yours * prior + (1 - prob_yours) * (1 - prior))
    )

    pred_yours = (prob_corrected >= 0.5).astype(int)
    acc_yours = np.mean(pred_yours == label_test)
    acc_yours_per.append(acc_yours)

    auc_yours = roc_auc_score(label_test, prob_corrected)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #
    

plt.figure()    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title("Testing Accuracy versus Training Size")
plt.tight_layout()
plt.savefig("figure4.png", dpi=300)


plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.title("Testing AUC versus Training Size")
plt.tight_layout()
plt.savefig("figure5.png", dpi=300)
plt.show()
