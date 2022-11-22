import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

df = p.read_csv('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Lab 3/week4a.csv')
c1 = df.iloc[:, 0]  # Reading in feature 1
c2 = df.iloc[:, 1] # Reading in feature 2
c3 = df.iloc[:, 2]  # Reading in target
#print(c3)

# i, a - Training data
plt.figure(0)  # Figure 1 for part A
plt.title("2 features plotted against each other") #Plot title
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'green', label = "y = -1") # legend for negative target data
y_pos_patch = mpatches.Patch(color = 'blue', label = "y = 1") # legend for positive target data
plt.legend(handles=[y_neg_patch, y_pos_patch], loc="upper right")  # Plot legend
colours = np.where(c3 == 1, 'blue', 'green')  #colors for target values
plt.scatter(c1, c2, c=colours) # Scatter plot
plt.show() #Show graphs

feature_cols = ['A', 'B']  # Columns with features we want to use
X = df[feature_cols]  # Assign features to variable X
y = c3  # Assign target to variable y


# i, a, i: Q range for polynomial feature (Cross val)
kf = KFold(n_splits=5)  #Splitting data up into 5 parts for cross valididation
mean_error=[]; std_error=[]  #Arrays for mean and standard deviation data
q_range = [1,2,3,4,5,6]  # Range of values for polynomial feature order
for q in q_range:
    Xpoly = PolynomialFeatures(q).fit_transform(X)  #Create polynomial featurez for given q
    model = LogisticRegression(penalty = 'l2') # Logistic regression model with l2 penalty
    temp=[]; # Array for storing values before being assigned into mean or stand deviation array
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])  #Fit thew model to training data
        ypred = model.predict(Xpoly[test])  #Use models to make predictions on test data
        temp.append(f1_score(y[test], ypred))  # Calculate f1 score for
    mean_error.append(np.array(temp).mean())  #Store f1 score
    std_error.append(np.array(temp).std()) # store standard deviation

plt.figure(1)
plt.title("F1 score Vs q (Polynomial features order)")  #Plot title
plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3, c = 'blue')  #Error bar graph
plt.xlabel('q value')  #X label
plt.ylabel('F1 score')  #Y label
f1_leg = mpatches.Patch(color = 'blue', label = "F1 score") # legend for f1 score
plt.legend(handles=[f1_leg], loc="center right")  # Plot legend
plt.show()

# i, a, ii: C range for Logistic regression (cross val)
Xpoly = PolynomialFeatures(2).fit_transform(X)  # Chose optimum q value
mean_error = []; std_error = [];
Ci_range = [0.000000005, 0.000005, .001, 0.1, 0.5, 1, 5]
ci_graph = []
for Ci in Ci_range:
    LR = LogisticRegression(penalty = 'l2', C = Ci) #LR model
    temp=[];

    kf = KFold(n_splits=5)
    for train, test in kf.split(Xpoly):  #Splitting test and train data
        LR.fit(Xpoly[train], y[train])  #fitting model
        ypred_LR = LR.predict(Xpoly[test])  #Predicting

        #temp.append(mean_squared_error(y[test],ypred_LR))
        temp.append(f1_score(y[test], ypred_LR))  # Calculate f1 score for
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        ci_graph.append(Ci)


plt.figure(2)
plt.title("F1 score Vs Ci (L2 penalty)") #Title
plt.errorbar(ci_graph,mean_error,yerr=std_error, c = 'blue')  # Error bar for cross val C values
plt.xlabel('Ci (L2 penalty)') #Y label
plt.ylabel('F1 score')  #y Label
f1_leg = mpatches.Patch(color = 'blue', label = "F1 score") # legend for f1 score
plt.legend(handles=[f1_leg], loc="center right")  # Plot legend
plt.show()

#Model predictions with tuned hyper paramters
model_log = LogisticRegression(penalty = 'l2', C = 1).fit(Xpoly, y)  # Logistic regression Model
y_pred_log = model_log.predict(Xpoly)  #Predictions

print("F1 score: ",f1_score(y, y_pred_log, average="macro")) #F1 score
df["D"] = y_pred_log  # Storing predictions in csv file

plt.figure(3)
plt.title("Logistic regression model with C = 1 and q = 2") #Title
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1") #legend
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1") #legend
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1") #legend
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1") #legend
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="center right") #Plotting legend
colours = np.where((df['D'] == 1) & (df['C'] == 1), 'blue', '#00000000')  #Color for positive correct prediction
colours1 = np.where((df['D'] == -1) & (df['C'] == 1), 'red', '#00000000') #Color for false negative
colours2 = np.where((df['D'] == 1) & (df['C'] == -1), 'orange', '#00000000') #Color for false negative
colours3 = np.where((df['D'] == -1) & (df['C'] == -1), 'green', '#00000000') #Color for negative correct prediction
plt.scatter(c1, c2, c=colours) #scatter for positve correct prediction
plt.scatter(c1, c2, c=colours1) #scatter for false negative
plt.scatter(c1, c2, c=colours2) #scatter for false positve
plt.scatter(c1, c2, c=colours3) #scatter for negative correct prediction
plt.show()

# i, b: K for KNN
mean_error=[]  #f1 score array
std_error=[]  #Standard deviation array
k_range = [1,3,5,7,9,11,13,15,17,19,21,23,25] # Range of k to try

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k).fit(Xpoly, y) #KNN model with varying

    scores = cross_val_score(model, Xpoly, y, cv=5, scoring='f1')  #F1 score used for classifier model
    mean_error.append(np.array(scores).mean())  #F1 score
    std_error.append(np.array(scores).std())  #Standard deviation


plt.figure(4) # Figure
plt.title("F1 score versus K for KNN")  # Plot title
plt.errorbar(k_range,mean_error,yerr=std_error,linewidth=3) # Error bar for cross val
plt.xlabel('K')  # X label
plt.ylabel('F1 Score')  # Y label
plt.show()  # Displaying graph

model_knn = KNeighborsClassifier(n_neighbors=23).fit(X, y)  # Optimised knn model
y_pred_knn = model_knn.predict(X)  # fitting knn model
df["E"] = y_pred_knn  # Storing knn predictions in csv file
print("F1 score: ",f1_score(y, y_pred_knn, average="macro")) #F1 score

plt.figure(5)  # Figure 1 for part A
plt.title("KNN model with K = 23") # Title
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1") #legend
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1") #legend
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1") #legend
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1") #legend
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="center right") #Plotting legend
colours = np.where((df['E'] == 1) & (df['C'] == 1), 'blue', '#00000000')  #Color for positive correct prediction
colours1 = np.where((df['E'] == -1) & (df['C'] == 1), 'red', '#00000000') #Color for false negative
colours2 = np.where((df['E'] == 1) & (df['C'] == -1), 'orange', '#00000000') #Color for false negative
colours3 = np.where((df['E'] == -1) & (df['C'] == -1), 'green', '#00000000') #Color for negative correct prediction
plt.scatter(c1, c2, c=colours) #scatter for positve correct prediction
plt.scatter(c1, c2, c=colours1) #scatter for false negative
plt.scatter(c1, c2, c=colours2) #scatter for false positve
plt.scatter(c1, c2, c=colours3) #scatter for negative correct prediction
plt.show()

conf_knn = confusion_matrix(y, y_pred_knn)  #Confusion matrix for knn
conf_log = confusion_matrix(y, y_pred_log)  # Confusion matrix for knn
print("Confusion matrix(KNN)", conf_knn)  #Printing confusion matrix
print("Confusion matrix(LR)", conf_log)  #Printing confusion matrix
model_bl = DummyRegressor(strategy = 'median')  # Baseline model
model_bl.fit(Xpoly, y) #Fitting baseline model
y_pred_bl = model_bl.predict(Xpoly)# Baseline model predictions
conf_bl = confusion_matrix(y, y_pred_bl)  # Confusion matrix for baseline model
print("Confusion matrix(BL)", conf_bl)  #Print baseline confusion matrix

plt.figure(6)
fpr, tpr, _ = roc_curve(y,model_log.decision_function(Xpoly))  # Defining false positives and true positives
plt.title("ROC curve for LR, KNN and Baseline classifiers") # Title
plt.plot(fpr,tpr, c = 'blue')  # Plot roc curve
plt.xlabel('False positive rate')  # X label
plt.ylabel('True positive rate')  # Y label

# calculate the fpr and tpr for all thresholds of the classification
probs = model_knn.predict_proba(X)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y, preds) #Knn fpr and tpr
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr,tpr, c = 'orange')  # Plot roc curve


y_neg_patch = mpatches.Patch(color = 'blue', label = "LR") #legend
y_pos_patch = mpatches.Patch(color = 'green', label = "Baseline") #legend
y_neg1_patch = mpatches.Patch(color = 'orange', label = "KNN") #legend
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch], loc="center right") #Plotting legend


fpr, tpr, thresholds = roc_curve(y, y_pred_bl) # baseline model
plt.plot(fpr,tpr, c = 'green')  # Plot roc curve
plt.show()

