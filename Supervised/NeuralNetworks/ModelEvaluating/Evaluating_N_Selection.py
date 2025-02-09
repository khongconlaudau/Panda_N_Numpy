import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import tensorflow as tf

import matplotlib.pyplot as plt

# Data set
x = np.array([1651.0000000000002, 1691.8163265306123, 1732.6326530612246, 1773.4489795918369, 1814.2653061224494, 1855.0816326530614, 1895.8979591836737, 1936.714285714286, 1977.530612244898, 2018.3469387755104, 2059.1632653061224, 2099.979591836735, 2140.795918367347, 2181.612244897959, 2222.4285714285716, 2263.244897959184, 2304.061224489796, 2344.877551020408, 2385.6938775510207, 2426.510204081633, 2467.3265306122453, 2508.1428571428573, 2548.95918367347, 2589.775510204082, 2630.591836734694, 2671.408163265306, 2712.2244897959185, 2753.0408163265306, 2793.8571428571427, 2834.673469387755, 2875.4897959183677, 2916.3061224489797, 2957.122448979592, 2997.9387755102043, 3038.7551020408164, 3079.571428571429, 3120.3877551020414, 3161.2040816326535, 3202.0204081632655, 3242.8367346938776, 3283.65306122449, 3324.469387755102, 3365.285714285714, 3406.1020408163267, 3446.918367346939, 3487.7346938775518, 3528.5510204081634, 3569.367346938776, 3610.1836734693875, 3651.0000000000005]).reshape(-1,1)
y = np.array([432.645217240638, 454.93552961965514, 471.52524757599184, 482.50638875194664, 468.35788633719716, 482.1525306782944, 540.0217555097203, 534.5842671578692, 558.346207611623, 566.4234447617514, 581.3976511525489, 596.4587372673582, 596.7148316883482, 619.4513901486196, 616.5762649903032, 653.1624598802053, 666.5199210989538, 670.5897593631339, 669.022887200185, 678.9093230505671, 707.436964393803, 710.7602881111783, 745.1913410872419, 729.8457618209563, 743.8029215960095, 738.2026910802764, 772.9461222927954, 772.2177055478061, 784.2138118483547, 776.4326081753612, 804.7762635147318, 833.2724588451545, 825.6903699737844, 821.0533253634433, 833.822037398165, 833.0614217591135, 825.6980744196746, 843.5773369341512, 869.3955170200721, 851.5030147483426, 863.1825574614074, 853.0060131500783, 877.1567537653887, 863.7419958565879, 874.6719415444281, 877.7363361538991, 874.1125977570639, 882.803295208019, 910.8327208115878, 897.4204630543912]).reshape(-1,1)

# Visualize the data set
# plt.scatter(x, y,marker='o', color='blue')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("input vs target")
# plt.show()


# get 60 % of the data set as training set and 40% remaining in tmp variables: x_, y_
# The random_state parameter in train_test_split ensures that the dataset is split the same way every time you run the code.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

#  Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test= train_test_split(x_, y_, test_size=0.50, random_state=1)

# del tmp variables
del x_, y_
if __name__ == '__main__':
    print(f"the shape of the training set (input)   is: {x_train.shape}")
    print(f"the shape of the training set (target) is: {y_train.shape}\n")
    print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
    print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
    print(f"the shape of the test set (input) is: {x_test.shape}")
    print(f"the shape of the test set (target) is: {y_test.shape}")


    # Initialize Standard Scaler class
    scaler_linear = StandardScaler()

    # Feature Scaling for x_train
    # z = (x - mean) / std

    X_train_scaled = scaler_linear.fit_transform(x_train)

    # Visualize the data after scaling
    # plt.scatter(X_train_scaled,y_train )
    # plt.show()
    print()
    print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Computed st deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

    # Train the model
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    # prediction
    yhat = linear_model.predict(X_train_scaled)

    '''
    Scikit-learn also has a built-in mean_squared_error() function that you can use.
    Take note though that as per the documentation,scikit-learn's implementation 
    only divides by m and not 2*m
    '''
    print(f"Training MSE: {mean_squared_error(yhat, y_train) / 2}")

    '''
    An important thing to note when using the z-score is you have to
    use the mean and standard deviation of the "training set" when scaling the cross validation set. 
    This is to ensure that your input features are transformed as expected by the model.
    '''
    X_cv_scaled = scaler_linear.transform(x_cv)
    print()
    print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

    yhat = linear_model.predict(X_cv_scaled)
    print(f"Cross validation MSE: {mean_squared_error(yhat, y_cv)/2} ")



    # Because cost function is too large we need to use different poly

    # Initialize Poly Features class with degree of 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    '''
    Note: The `e+<number>` in the output denotes how many places the decimal point should 
    be moved. For example, `3.24e+03` is equal to `3240`
    '''
    print()
    print(X_train_mapped[:5])

    # Scaling Featuring
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    print()
    print("After scaling: \n",X_train_mapped_scaled[:5])

    # Training model
    model = LinearRegression()

    model.fit(X_train_mapped_scaled, y_train)
    yhat  = model.predict(X_train_mapped_scaled)
    print()
    print(f"Training MSE: {mean_squared_error(yhat, y_train) / 2}")

    # Add the polynomial features to the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    # Scale
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute CV MSE
    yhat = model.predict(X_cv_mapped_scaled)
    print(f"Cross validation MSE: {mean_squared_error(yhat, y_cv) / 2}")


    '''
    We can loop over ~ 10 times to choose the best poly
    '''

    train_mses, cv_mses, models, polys, scalers = [], [], [], [], []

    for degree in range(1,11):
        # Add popy features to the traning set
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)
        polys.append(poly)

        # add scaling featuring
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # create training model
        model = LinearRegression()
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # compute training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mses.append(mean_squared_error(yhat, y_train) / 2)

        # add poly and scale featuring to cv set
        X_cv_mapped = poly.transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # compute cv  MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mses.append(mean_squared_error(yhat, y_cv) / 2)

    degree = range(1,11)
    # Visualize the data
    # plt.plot(degree, train_mses, label='Training MSE',color='red')
    # plt.plot(degree, cv_mses, label='Cross Validation MSE',color='blue')
    # plt.xlabel('degree')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.title('degree of polynomial vs. train and CV MSEs')
    # plt.show()

    # Choosing the best model
    # Get the model with the lowest CV MSE (add 1 because list indices start at 0)
    b_degree = np.argmin(cv_mses)
    print(f"Lowest CV MSE is found in the model with degree={b_degree+1}")


    # Add pylo and scaling features to x_test

    X_test_mapped = polys[b_degree].transform(x_test)
    X_test_mapped_scaled = scalers[b_degree].transform(X_test_mapped)

    # Compute MSE of x_test
    yhat = models[b_degree].predict(X_test_mapped_scaled)
    test_mse = mean_squared_error(yhat, y_test)/2
    print()
    print(f"Training MSE: {train_mses[b_degree]:.2f}")
    print(f"Cross Validation MSE: {cv_mses[b_degree]:.2f}")
    print(f"Test MSE: {test_mse:.2f}")

