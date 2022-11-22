import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.dummy import DummyRegressor

df = p.read_csv('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Lab 2/week3.csv')
c1 = df.iloc[:, 0]  # Reading in feature 1
c2 = df.iloc[:, 1] # Reading in feature 2
c3 = df.iloc[:, 2]  # Reading in target
print("CSV file read in")

#Plotting features
fig = plt.figure(0)  # Figure 1 for part A
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Features X1 and X2 plotted against the target y") # Title of the plot4
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')  # Z label
ax.scatter(c1, c2, c3, label = 'Features plotted against target y')  #Plotting th scatter
leg = ax.legend()
plt.show()  # Showing the scatter
print("finished showing graph")

feature_cols = ['A', 'B']  # Columns with features we want to use
X = df[feature_cols]  # Assign features to variable X
y = c3  # Assign target to variable y

#Creating the polynomial features
power5_features = PolynomialFeatures(5).fit_transform(X)  # Power of 5 features
print(power5_features[0])
print("Features generated")

lasso1 = Lasso(alpha=.5, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Lasso model 1 (C = 1) coefficients\n", lasso1.coef_)  # Printing model coefficients
print("Lasso model 1 (C = 1) intercept", lasso1.intercept_)  # Printing model intercept

lasso2 = Lasso(alpha=.05, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Lasso model 2 (C = 10) coefficients\n", lasso2.coef_)  # Printing model coefficients
print("Lasso model 2 (C = 10) intercept", lasso2.intercept_)  # Printing model intercept

lasso3 = Lasso(alpha=.0005, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Lasso model 3 (C = 1000) coefficients\n", lasso3.coef_)  # Printing model coefficients
print("Lasso model 3 (C = 1000) intercept", lasso3.intercept_)  # Printing model intercept

#Creating test features
X_test = []

grid = np.linspace(-5, 5, num=10)
for i in grid:
    for j in grid:
        X_test.append([i,j])
X_test = np.array(X_test)
#print(X_test)

power5_features_test = PolynomialFeatures(5).fit_transform(X_test)  # Power of 5 features
y_lassopredict1 = lasso1.predict(power5_features_test)  # Using lasso model to predict values
#print("Lasso model 1 predictions", y_lassopredict1)
y_lassopredict2 = lasso2.predict(power5_features_test)  # Using lasso model to predict values
#print("Lasso model 2 predictions", y_lassopredict2)
y_lassopredict3 = lasso3.predict(power5_features_test)  # Using lasso model to predict values
#print("Lasso model 3 predictions", y_lassopredict3)

# Displaying the predictions with the test features for each lasso model
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Lasso model 1 (C = 1)")
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')
ax.legend(["blue", "green"])
ax.scatter(X_test[:,0], X_test[:,1], y_lassopredict1, color = 'r', label = 'Lasso model (C = 1)')
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')
leg = ax.legend()
plt.show()

#plt.figure(3)
fig = plt.figure(2)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Lasso model 2 (C = 10)")  # Plot title
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (y)')  # Z label
ax.scatter(X_test[:,0], X_test[:,1], y_lassopredict2, color = 'r', label = 'Lasso model (C = 10)')  # Plotting scatter for predictions
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')  #Plotting original features
leg = ax.legend()
plt.show()

#plt.figure(4)
fig = plt.figure(3)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Lasso model 3 (C = 1000)")
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')
ax.legend(["blue", "green"], loc ="lower right")
ax.scatter(X_test[:,0], X_test[:,1], y_lassopredict3, color = 'r', label = 'Lasso model (C = 1000)')
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')
leg = ax.legend()
plt.show()

lasso3 = Ridge(alpha=10000, max_iter=10000).fit(power5_features, y)
# Ridge
ridge1 = Ridge(alpha=1000000, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Ridge model 1 (C = 0.0000005) coefficients\n", ridge1.coef_)  # Printing model coefficients
print("Ridge model 1 (C = .0000005) intercept", ridge1.intercept_)  # Printing model intercept

ridge2 = Ridge(alpha=100000, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Ridge model 2 (C = .000005) coefficients\n", ridge2.coef_)  # Printing model coefficients
print("Ridge model 2 (C = .000005) intercept", ridge2.intercept_)  # Printing model intercept

ridge3 = Ridge(alpha=.05, max_iter=10000).fit(power5_features, y)  #Creating lasso model with X and y
print("Ridge model 3 (C = 1) coefficients\n", ridge3.coef_)  # Printing model coefficients
print("Ridge model 3 (C = 1) intercept", ridge3.intercept_)  # Printing model intercept

y_ridgepredict1 = ridge1.predict(power5_features_test)  # Using lasso model to predict values
#print("Ridge model 1 predictions", y_ridgepredict1)
y_ridgepredict2 = ridge2.predict(power5_features_test)  # Using lasso model to predict values
#print("Ridge model 2 predictions", y_ridgepredict2)
y_ridgepredict3 = ridge3.predict(power5_features_test)  # Using lasso model to predict values
#print("Ridge model 3 predictions", y_ridgepredict3)

fig = plt.figure(4)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Ridge model 1 (C = .0000005)")
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')
ax.legend(["blue", "green"], loc = "lower right")
ax.scatter(X_test[:,0], X_test[:,1], y_ridgepredict1, color = 'r', label = 'Ridge model (C = .0000005)')
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')
leg = ax.legend()
plt.show()

fig = plt.figure(5)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Ridge model 2 (C = .000005)")
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')
ax.legend(["blue", "green"], loc = "lower right")
ax.scatter(X_test[:,0], X_test[:,1], y_ridgepredict2, color = 'r', label = 'Ridge model (C = .000005)')
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')
leg = ax.legend()
plt.show()

fig = plt.figure(6)
ax = fig.add_subplot(projection='3d') #Plotting
plt.title("Ridge model 3 (C = 1)")
plt.xlabel("1st feature (X1)")  # X label
plt.ylabel("2nd Feature (X2)")  # Y label
ax.set_zlabel('Target (Y)')
ax.legend(["blue", "green"], loc ="lower right")
ax.scatter(X_test[:,0], X_test[:,1], y_ridgepredict3, color = 'r', label = 'Ridge model (C = 1)')
ax.scatter(c1, c2, c3, color = 'b', label = 'Original features')
leg = ax.legend()
plt.show()


mean_error=[]; std_error=[]; mean_error_dum = []; std_error_dum = []  # Arrays for error and standard deviation for baseline and model
Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100]  # Array for range of Ci
ci_graph = []
for Ci in Ci_range:
    cross_fold = Lasso(alpha=1/(2*Ci))  #Model
    dummy_regr = DummyRegressor(strategy="mean")  #Baseline model
    temp=[]; temp_dum = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)  # Cross validation splitting data into 5 equal segments
    for train, test in kf.split(power5_features):
        cross_fold.fit(power5_features[train], y[train])  #Fitting to training data using lasso
        dummy_regr.fit(power5_features[train], y[train])  # Fitting to training data using baseline
        ypred = cross_fold.predict(power5_features[test]) # Predicting test data using lasso
        y_pred_dummy = dummy_regr.predict(power5_features[test])  #Predicting test datausing baseline
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
        mean_error.append(np.array(temp).mean()) # Adding mean value for ci to mean array for lasso
        std_error.append(np.array(temp).std())  # Adding standard deviation value for ci to mean array for lasso

        temp_dum.append(mean_squared_error(y[test], y_pred_dummy))
        mean_error_dum.append(np.array(temp_dum).mean())  # Adding standard deviation value for ci to mean array for baseline
        std_error_dum.append(np.array(temp_dum).std())  # Adding standard deviation value for ci to mean array for lasso baseline

        ci_graph.append(Ci)
        #print(ci_graph)
print("MSE: ", mean_error)
import matplotlib.pyplot as plt
ax = plt.figure(7)
plt.title("Mean versus C for the Lasso model")  #Title
plt.errorbar(ci_graph,mean_error,yerr=std_error, label = 'Mean versus C for Lasso model')  # Error bar graph for lasso
plt.errorbar(ci_graph,mean_error_dum,yerr=std_error_dum, label = 'Mean versus C for baseline')  # Error bar graph for lasso
plt.xlabel('Ci')  # Xlabel
plt.ylabel('Mean square error')  #Ylabel
plt.xlim((0,50))  # Axis scale
leg = ax.legend(loc = 'center right') #Legend
plt.show()  #Dispaly

mean_error = []; std_error = []; mean_error_dum = []; std_error_dum = []
Ci_range = [0.000000005, 0.000005, .001, 0.1, 0.5, 1, 5]
ci_graph = []
for Ci in Ci_range:
    cross_fold = Ridge(alpha=1/(2*Ci))
    dummy_regr = DummyRegressor(strategy="mean")
    temp=[]; temp_dum = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for train, test in kf.split(power5_features):
        cross_fold.fit(power5_features[train], y[train])
        dummy_regr.fit(power5_features[train], y[train])
        ypred = cross_fold.predict(power5_features[test])
        y_pred_dummy = dummy_regr.predict(power5_features[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        temp_dum.append(mean_squared_error(y[test], y_pred_dummy))
        mean_error_dum.append(np.array(temp_dum).mean())
        std_error_dum.append(np.array(temp_dum).std())

        ci_graph.append(Ci)
        #print(ci_graph)
print("MSE: ", mean_error)
import matplotlib.pyplot as plt
ax = plt.figure(8)
plt.title("Mean versus C for the Ridge model")
plt.errorbar(ci_graph,mean_error,yerr=std_error, label = 'Mean versus C Ridge model')
plt.errorbar(ci_graph,mean_error_dum,yerr=std_error_dum, label = 'Mean versus C for baseline model')
plt.xlabel('Ci')
plt.ylabel('Mean square error')
plt.xlim((0,5))
leg = ax.legend(loc = 'center right')
plt.show()






