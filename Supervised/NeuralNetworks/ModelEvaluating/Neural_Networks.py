import numpy as np
from Tools.scripts.cleanfuture import verbose

from Evaluating_N_Selection import x_train, y_train, x_cv, y_cv, x_test, y_test
from utils.custom_funtion import build_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
# Add poly n scaling features with default 1
degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)




# Initialize lists that will contain the errors for each model
nn_train_mses = []
nn_cv_mses = []

# build models
nn_models = build_model()


for model in nn_models:

    # Setup loss and optimizer
    model.compile(loss=tf.keras.losses.MSE,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),)

    print(f"Training: {model.name}")
    # Train the model
    '''
    verbose=0: No output is printed to the console during training.
    Use this if you want a completely silent training process.
    
    verbose=1: Progress bar output is printed to the console.
    This is the default setting.
    Displays the progress of each epoch with a progress bar, loss, and other metrics.
    '''
    model.fit(X_train_mapped_scaled,y_train,epochs=300, verbose=0)

    print("Done!\n")

    yhat = model.predict(X_train_mapped_scaled)
    nn_train_mses.append(mean_squared_error(y_train, yhat)/2)

    # Record the cross validation MSEs
    yhat2 = model.predict(X_cv_mapped_scaled)
    nn_cv_mses.append(mean_squared_error(y_cv, yhat2)/2)


# Results
print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
        )

b_model = np.argmin(nn_cv_mses)

yhat = nn_models[b_model].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) /2

print(f"Selected Model: {b_model}")
print(f"Training MSE: {nn_train_mses[b_model]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[b_model]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
