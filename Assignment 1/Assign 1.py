# Libraries needed
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.preprocessing import maxabs_scale
import matplotlib.patches as mpatches

df = p.read_csv('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Lab 1/week2csv.csv')
# print("CSV data\n", df)  # Print CSV file read in
c1 = df.iloc[:, 0]  # Reading in feature 1
c2 = df.iloc[:, 1]  # Reading in feature 2
c3 = df.iloc[:, 2]  # Reading in target
c4 = df.iloc[:, 3]  # Reading in feature 3
c5 = df.iloc[:, 4]  # Reading in feature 4

plt.figure(0)  # Figure 1 for part A
plt.title("2 features plotted against each other")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'green', label = "y = -1") # legend for negative target data
y_pos_patch = mpatches.Patch(color = 'blue', label = "y = 1") # egend for positive target data
plt.legend(handles=[y_neg_patch, y_pos_patch], loc="upper right")
colours = np.where(df['C'] == 1, 'blue', 'green')  #colors for target values
plt.scatter(c1, c2, c=colours, label="+1 = blue\n -1 = green") # Scatter plot
plt.show()

# Part A
feature_cols = ['A', 'B']  # Columns with features we want to use
X = df[feature_cols]  # Assign features
y = c3  # Target

model = LogisticRegression(penalty='none', solver='lbfgs')  # Logistic regression model
model.fit(X, y)  # Fitting logistic regression model
y_pred = model.predict(X)  # Predicting target variable using model
df["F"] = y_pred  # Storing predictions in csv file
#print("Appended CSV: \n", df) # Printing modified CSV file

print("Model coefficients: ", model.coef_)  # Model coefficients
print("Model intercept:", (model.intercept_))  # Model intercept
print("Model predictions: ", y_pred)  # Print out predictions for logistic regression model
print("Accuracy: ", model.score(X, y))  # Accuracy
print("F1 score: ",f1_score(y, y_pred, average="macro")) #F1 score
print("Precision: ",precision_score(y, y_pred, average="macro")) #Precison
print("Recall: ", recall_score(y, y_pred, average="macro")) #Recall
conf = confusion_matrix(y, y_pred)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf)  # Print confusion matrix

weight_0_log_boundary = model.intercept_  # Model intercept
weight_1_log_boundary = model.coef_[0][0]  # 1st model coefficent
weight_2_log_boundary = model.coef_[0][1]  # 2nd model coefficient
c_log_boundary = -(weight_0_log_boundary)/weight_2_log_boundary  # Y axis intercept
m_log_boundary = -(weight_1_log_boundary)/weight_2_log_boundary  # X axis intercept
x_log_boundary = np.array([-1, 1])  # X values for equation
log_decision_boundary = m_log_boundary*x_log_boundary + c_log_boundary  # Y=mx+c
plt.plot(x_log_boundary, log_decision_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.figure(1)  # Figure 1 for part A
plt.title("Logistic regression model") # Title
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1") #legend
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1") #legend
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1") #legend
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1") #legend
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right") #Plotting legend
colours = np.where((df['F'] == 1) & (df['C'] == 1), 'blue', '#00000000')  #Color for positive correct prediction
colours1 = np.where((df['F'] == -1) & (df['C'] == 1), 'red', '#00000000') #Color for false negative
colours2 = np.where((df['F'] == 1) & (df['C'] == -1), 'orange', '#00000000') #Color for false negative
colours3 = np.where((df['F'] == -1) & (df['C'] == -1), 'green', '#00000000') #Color for negative correct prediction
plt.scatter(c1, c2, c=colours) #scatter for positve correct prediction
plt.scatter(c1, c2, c=colours1) #scatter for false negative
plt.scatter(c1, c2, c=colours2) #scatter for false positve
plt.scatter(c1, c2, c=colours3) #scatter for negative correct prediction
plt.show()

# Part B
#SVM models with varying c values
svmmodel0 = LinearSVC(C=0.00001).fit(X, y)
svmmodel1 = LinearSVC(C=0.001).fit(X, y)
svmmodel2 = LinearSVC(C=1.0).fit(X, y)
svmmodel3 = LinearSVC(C=100).fit(X, y)
svmmodel4 = LinearSVC(C=10000).fit(X, y)

#Fitting model and using them to predict
svmmodel0.fit(X, y)
y_pred_svm0 = svmmodel0.predict(X)
svmmodel1.fit(X, y)
y_pred_svm1 = svmmodel1.predict(X)
svmmodel2.fit(X, y)
y_pred_svm2 = svmmodel2.predict(X)
svmmodel3.fit(X, y)
y_pred_svm3 = svmmodel3.predict(X)
svmmodel4.fit(X, y)
y_pred_svm4 = svmmodel4.predict(X)


df["G"] = y_pred_svm1  # Storing predictions in csv file
df["H"] = y_pred_svm2  # Storing predictions in csv file
df["I"] = y_pred_svm3  # Storing predictions in csv file
df["J"] = y_pred_svm0  # Storing predictions in csv file
df["K"] = y_pred_svm4  # Storing predictions in csv file

# printing predictions
print("SVM predictions for C = 0.00001 \n", y_pred_svm0)
print("SVM predictions for C = 0.001 \n", y_pred_svm1)
print("SVM predictions for C = 1 \n", y_pred_svm2)
print("SVM predictions for C = 100 \n", y_pred_svm3)
print("SVM predictions for C = 10000 \n", y_pred_svm4)
#printing coefficients
print("SVM model 0 coefficients", svmmodel0.coef_)
print("SVM model 1 coefficients", svmmodel1.coef_)
print("SVM model 2 coefficients", svmmodel2.coef_)
print("SVM model 3 coefficients", svmmodel3.coef_)
print("SVM model 4 coefficients", svmmodel4.coef_)

print("SVM model 0 intercept", svmmodel0.intercept_)
print("SVM model 1 intercept", svmmodel1.intercept_)
print("SVM model 2 intercept", svmmodel2.intercept_)
print("SVM model 3 intercept", svmmodel3.intercept_)
print("SVM model 4 intercept", svmmodel4.intercept_)
#printing performance parameters
print("Accuracy of SVM with C = .00001:  ", svmmodel0.score(X, y))  # Accuracy
print("F1 score of SVM with C = .00001: ",f1_score(y, y_pred_svm0, average="macro"))
print("Precision of SVM with C = .00001: ",precision_score(y, y_pred_svm0, average="macro"))
print("Recall of SVM with C = .00001: ", recall_score(y, y_pred_svm0, average="macro"))

print("Accuracy of SVM with C = 0.001: ", svmmodel1.score(X, y))  # Accuracy
print("F1 score of SVM with C = 0.001: ", f1_score(y, y_pred_svm1, average="macro"))
print("Precision of SVM with C = 0.001: ",precision_score(y, y_pred_svm1, average="macro"))
print("Recall of SVM with C = 0.001: : ", recall_score(y, y_pred_svm1, average="macro"))

print("Accuracy of SVM with C = 1:  ", svmmodel2.score(X, y))  # Accuracy
print("F1 score of SVM with C = 1: ",f1_score(y, y_pred_svm2, average="macro"))
print("Precision of SVM with C = 1: ",precision_score(y, y_pred_svm2, average="macro"))
print("Recall of SVM with C = 1: ", recall_score(y, y_pred_svm2, average="macro"))

print("Accuracy of SVM with C = 100:  ", svmmodel3.score(X, y))  # Accuracy
print("F1 score of SVM with C = 100: ",f1_score(y, y_pred_svm3, average="macro"))
print("Precision of SVM with C = 100: ",precision_score(y, y_pred_svm3, average="macro"))
print("Recall of SVM with C = 100: ", recall_score(y, y_pred_svm3, average="macro"))

print("Accuracy of SVM with C = 10000:  ", svmmodel4.score(X, y))  # Accuracy
print("F1 score of SVM with C = 10000: ",f1_score(y, y_pred_svm4, average="macro"))
print("Precision of SVM with C = 10000: ",precision_score(y, y_pred_svm4, average="macro"))
print("Recall of SVM with C = 10000: ", recall_score(y, y_pred_svm4, average="macro"))


conf_svm0 = confusion_matrix(y, y_pred_svm0)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf_svm0)  # Print confusion matrix
conf_svm1 = confusion_matrix(y, y_pred_svm1)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf_svm1)  # Print confusion matrix
conf_svm2 = confusion_matrix(y, y_pred_svm2)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf_svm2)  # Print confusion matrix
conf_svm3 = confusion_matrix(y, y_pred_svm3)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf_svm3)  # Print confusion matrix
conf_svm4 = confusion_matrix(y, y_pred_svm4)  # Confusion matrix for true, false positive and negatives
print("confusion matrix: \n", conf_svm4)  # Print confusion matrix

plt.figure(6)  # Figure 1 for part A
weight_0_svm0 = svmmodel0.intercept_  # Model intercept
weight_1_svm0 = svmmodel0.coef_[0][0]  # 1st model coefficent
weight_2_svm0 = svmmodel0.coef_[0][1]  # 2nd model coefficient
c_svm0 = -(weight_0_svm0)/weight_2_svm0  # Y axis intercept
m_svm1 = -(weight_1_svm0)/weight_2_svm0  # X axis intercept
x_svm0 = np.array([-1, 1])  # X values for equation
svm0_boundary = m_svm1*x_svm0 + c_svm0  # Y=mx+c
plt.plot(x_svm0, svm0_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.title("SVM with C = 0.00001")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['J'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['J'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['J'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['J'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()

plt.figure(2)  # Figure 1 for part A
weight_0_svm1 = svmmodel1.intercept_  # Model intercept
weight_1_svm1 = svmmodel1.coef_[0][0]  # 1st model coefficent
weight_2_svm1 = svmmodel1.coef_[0][1]  # 2nd model coefficient
c_svm1 = -(weight_0_svm1)/weight_2_svm1  # Y axis intercept
m_svm1 = -(weight_1_svm1)/weight_2_svm1  # X axis intercept
x_svm1 = np.array([-1, 1])  # X values for equation
svm1_boundary = m_svm1*x_svm1 + c_svm1  # Y=mx+c
plt.plot(x_svm1, svm1_boundary, linewidth=4, color='y')  # Plotting decision boundary

#plt.rc('font', size=12)
plt.title("SVM with C = 0.001")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['G'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['G'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['G'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['G'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()



plt.figure(3)  # Figure 1 for part A
weight_0_svm2 = svmmodel2.intercept_  # Model intercept
weight_1_svm2 = svmmodel2.coef_[0][0]  # 1st model coefficent
weight_2_svm2 = svmmodel2.coef_[0][1]  # 2nd model coefficient
c_svm2 = -(weight_0_svm2)/weight_2_svm2  # Y axis intercept
m_svm2 = -(weight_1_svm2)/weight_2_svm2  # X axis intercept
x_svm2 = np.array([-1, 1])  # X values for equation
svm2_boundary = m_svm2*x_svm2 + c_svm2  # Y=mx+c
plt.plot(x_svm1, svm2_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.title("SVM with C = 1")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['H'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['H'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['H'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['H'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()



plt.figure(4)  # Figure 1 for part A
weight_0_svm3 = svmmodel3.intercept_  # Model intercept
weight_1_svm3 = svmmodel3.coef_[0][0]  # 1st model coefficent
weight_2_svm3 = svmmodel3.coef_[0][1]  # 2nd model coefficient
c_svm3 = -(weight_0_svm3)/weight_2_svm3  # Y axis intercept
m_svm3 = -(weight_1_svm3)/weight_2_svm3  # X axis intercept
x_svm3 = np.array([-1, 1])  # X values for equation
svm3_boundary = m_svm3*x_svm3 + c_svm3  # Y=mx+c
plt.plot(x_svm3, svm3_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.title("SVM with C = 100")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['I'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['I'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['I'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['I'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()

plt.figure(7)  # Figure 1 for part A
weight_0_svm4 = svmmodel4.intercept_  # Model intercept
weight_1_svm4 = svmmodel4.coef_[0][0]  # 1st model coefficent
weight_2_svm4 = svmmodel4.coef_[0][1]  # 2nd model coefficient
c_svm4 = -(weight_0_svm4)/weight_2_svm4  # Y axis intercept
m_svm4 = -(weight_1_svm4)/weight_2_svm4  # X axis intercept
x_svm4 = np.array([-1, 1])  # X values for equation
svm4_boundary = m_svm4*x_svm4 + c_svm4  # Y=mx+c
plt.plot(x_svm4, svm4_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.title("SVM with C = 10000")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['K'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['K'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['K'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['K'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()

#Part C
feature_cols_4feature = ['A', 'B', 'D', 'E']

X_4feature = df[feature_cols_4feature]
print(X_4feature)
y_4feature = c3

model_4feature = LogisticRegression(penalty='none', solver='lbfgs')
model_4feature.fit(X_4feature, y_4feature)
y_pred_4feature = model_4feature.predict(X_4feature)
df["L"] = y_pred_4feature
print("Appended CSV: \n", df)

print("Model coefficients (4 features): ", model_4feature.coef_)
print("Model intercept (4 features):", (model_4feature.intercept_))
print("Model predictions (4 features): ", y_pred_4feature)
print("Accuracy (4 features): \n", model_4feature.score(X_4feature, y_4feature))  # Accuracy
print("F1 score (4 features): ",f1_score(y, y_pred_4feature, average="macro"))
print("Precision (4 features): ",precision_score(y, y_pred_4feature, average="macro"))
print("Recall (4 features): ", recall_score(y, y_pred_4feature, average="macro"))
conf_4feature = confusion_matrix(y, y_pred_4feature)
print("confusion matrix (4 features): \n", conf_4feature)

plt.figure(5)
weight_0_4feat = model_4feature.intercept_  # Model intercept
weight_1_4feat = model_4feature.coef_[0][0]  # 1st model coefficent
weight_2_4feat = model_4feature.coef_[0][1]  # 2nd model coefficient
c_4feat= -(weight_0_4feat)/weight_2_4feat  # Y axis intercept
m_4feat = -(weight_1_4feat)/weight_2_4feat  # X axis intercept
x_4feat = np.array([-1, 1])  # X values for equation
fourfeat_boundary = m_4feat*x_4feat + c_4feat  # Y=mx+c
#plt.plot(x_4feat, fourfeat_boundary, linewidth=4, color='y')  # Plotting decision boundary

plt.title("Logistic regression with 4 features")
plt.xlabel("1st feature")  # X label
plt.ylabel("2nd Feature")  # Y label
y_neg_patch = mpatches.Patch(color = 'blue', label = "y = 1, y^ = 1")
y_pos_patch = mpatches.Patch(color = 'red', label = "y = 1, y^ = -1")
y_neg1_patch = mpatches.Patch(color = 'orange', label = "y = -1, y^ = 1")
y_pos1_patch = mpatches.Patch(color = 'green', label = "y = -1, y^ = -1")
plt.legend(handles=[y_neg_patch, y_pos_patch, y_neg1_patch, y_pos1_patch], loc="upper right")
colours = np.where((df['L'] == 1) & (df['C'] == 1), 'blue', '#00000000')
colours1 = np.where((df['L'] == -1) & (df['C'] == 1), 'red', '#00000000')
colours2 = np.where((df['L'] == 1) & (df['C'] == -1), 'orange', '#00000000')
colours3 = np.where((df['L'] == -1) & (df['C'] == -1), 'green', '#00000000')
plt.scatter(c1, c2, c=colours)
plt.scatter(c1, c2, c=colours1)
plt.scatter(c1, c2, c=colours2)
plt.scatter(c1, c2, c=colours3)
plt.show()







