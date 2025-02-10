import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
RANDOM_STATE = 55

# Load data using panda
df = pd.read_csv("15-heart.csv")

# One-hot encoding using Pandas
cat_variables = ['Sex', 'ChestPainType','RestingECG','ExerciseAngina','ST_Slope']


# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data= df, columns=cat_variables, prefix=cat_variables)

features = [x for x in df.columns if x not in 'HeartDisease'] # remove target values

# Splitting the dataset
# help(train_test_split)
'''
random_state : int, RandomState instance or None, default=None
Controls the shuffling applied to the data before applying the split.
Pass an int for reproducible output across multiple function calls.
'''

X_train, X_val, y_train, y_val = train_test_split(df[features],df['HeartDisease'],train_size=0.8, random_state=RANDOM_STATE)

print(f"train samples: {len(X_train)}")
print(f"validation samples: {len(X_val)}")
print(f"target portion samples: {sum(y_train)/len(y_train):.4f}")


# mean_sample_splits: sets a threshold:
# If a node has fewer than min_samples_split samples, the algorithm does not split it further.
min_sample_splits_list = [2, 10, 30, 50, 100, 200, 300, 700]
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

accuracy_list_train = []
accuracy_list_val = []

for min_sample_split in min_sample_splits_list:
    model = DecisionTreeClassifier(min_samples_split = min_sample_split,
                                   random_state=RANDOM_STATE).fit(X_train,y_train)

    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)

    accuracy_list_train.append(accuracy_score(y_train, predictions_train))
    accuracy_list_val.append(accuracy_score(y_val, predictions_val))

# plt.title('Train x Validation metrics')
# plt.xlabel('min_samples_split')
# plt.ylabel('accuracy')
# plt.xticks(ticks= range(len(min_sample_splits_list)), labels=min_sample_splits_list)
# plt.plot(accuracy_list_train)
# plt.plot(accuracy_list_val)
# plt.legend(['Train', 'Validation'])
# plt.show()

'''
We can see that from 10 to 30, and from 30 to 50, even though it does not improve the validation accuracy, 
it brings the training accuracy closer to it, showing a reduction in overfitting.
'''

# accuracy_list_train = []
# accuracy_list_val = []
# for max_depth in max_depth_list:
#     model = DecisionTreeClassifier(max_depth=max_depth,random_state=RANDOM_STATE).fit(X_train,y_train)
#     predictions_train = model.predict(X_train)
#     predictions_val = model.predict(X_val)
#     accuracy_list_train.append(accuracy_score(y_train, predictions_train))
#     accuracy_list_val.append(accuracy_score(y_val, predictions_val))
#
# plt.title('Train x Validation metrics')
# plt.xlabel('max_depth')
# plt.ylabel('accuracy')
# plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
# plt.plot(accuracy_list_train)
# plt.plot(accuracy_list_val)
# plt.legend(['Train','Validation'])
# plt.show()

'''We can see that reducing max depth can reduce overfitting'''


'''
So we can choose the best values for these two hyper-parameters for our model to be:
max_depth = 4
min_samples_split = 50
'''


decision_tree_model = DecisionTreeClassifier(min_samples_split=50, max_depth=4, random_state=RANDOM_STATE).fit(X_train,y_train)
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")


# XGBoost
n = int(len(X_train)*0.8) # use 80% for train and 20% for eval
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

'''
early_stopping_rounds: This parameter helps to stop the model training 
if its evaluation metric is no longer improving on the validation set. It's set to 10.

Each successive round's evaluation metric is compared to the best metric. If the model 
goes 10 rounds where none have a better metric than the best one, then the model stops training.

he model is returned at its last state when training terminated, not its state during the best round. For example, if the model stops at round 26, 
but the best round was 16, the model's training state at round 26 is returned, not round 16.
'''

xgb_model = XGBClassifier(n_estimators=500,learning_rate=0.1,verbosity=0,random_state=RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set=[(X_train_eval,y_train_eval)],early_stopping_rounds=10)
xgb_model.best_iteration()
