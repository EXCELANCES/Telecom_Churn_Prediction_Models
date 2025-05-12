"""
authors: Ertugrul Asliyuce
"""
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,recall_score, precision_score, f1_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV    
from sklearn.preprocessing import StandardScaler
import os

# EDA

file_path = os.path.dirname(os.path.abspath(__file__))
file_path = file_path.replace("\\", "/")

# Load the dataset
data_raw = pd.read_csv( file_path + '/Telecom_Churn_Data.csv')

data = data_raw.copy()
#change space character with Null to count
columns=data.columns
for col in columns:
    data[col]=data[col].apply(lambda x: np.nan if x==" " else x)

data["Partner"]=data["Partner"].apply(lambda x: 1 if x=="Yes" else 0)
data["Dependents"]=data["Dependents"].apply(lambda x: 1 if x=="Yes" else 0)
data["PhoneService"]=data["PhoneService"].apply(lambda x: 1 if x=="Yes" else 0)
data["PaperlessBilling"]=data["PaperlessBilling"].apply(lambda x: 1 if x=="Yes" else 0)
data["Churn"]=data["Churn"].apply(lambda x: 1 if x=="Yes" else 0)

print("Dataset Size:", data.shape)
print("Data Types:\n", data.dtypes)  
print("-----------------------------------------------------------------")   

#Convert SeniorCitizen column as category
data["gender"]=data["gender"].astype("category")
data["SeniorCitizen"]=data["SeniorCitizen"].astype("bool")
data["Partner"]=data["Partner"].astype("bool")
data["Dependents"]=data["Dependents"].astype("bool")
data["PhoneService"]=data["PhoneService"].astype("bool")
data["MultipleLines"]=data["MultipleLines"].astype("category")
data["InternetService"]=data["InternetService"].astype("category")
data["OnlineSecurity"]=data["OnlineSecurity"].astype("category")
data["OnlineBackup"]=data["OnlineBackup"].astype("category")
data["DeviceProtection"]=data["DeviceProtection"].astype("category")
data["TechSupport"]=data["TechSupport"].astype("category")
data["StreamingTV"]=data["StreamingTV"].astype("category")
data["Contract"]=data["Contract"].astype("category")
data["PaperlessBilling"]=data["PaperlessBilling"].astype("bool")
data["PaymentMethod"]=data["PaymentMethod"].astype("category")
data["TotalCharges"]=data["TotalCharges"].astype("float")
data["Churn"]=data["Churn"].astype("bool")


print("Missing Values:")
print()
print(data.isnull().sum())
print("-----------------------------------------------------------------")  

#get null counts
null_counts = data.isnull().sum()
#percentage of null data
null_percentage = (null_counts / len(data))*100

formatted_null_percentage = null_percentage.apply(lambda x: f"%{x:.2f}")
print(formatted_null_percentage)
print("-----------------------------------------------------------------") 

#Null data proportion is less than %5 so we can remove them
data = data.dropna()
# See results
print("Null-Free Data:")
print(data)
print("-----------------------------------------------------------------")  

#Unique values of Dataset's Categorical Values(its because is there a variable which is different from has to be)
columns=data.columns[:-3] 
for col in columns:
    print(f"Unique Values for {col} Column:")
    print(data[col].unique())
    print("Count:",data[col].nunique())
    print()

print("Unique Values for Churn Column:")
print(data["Churn"].unique())
print("Count:",data["Churn"].nunique())
print()

#This is seperator for not to confuse codes
print("-----------------------------------------------------------------")  

print(data.info())
print(data.describe().T)

print("-----------------------------------------------------------------")  

# Numberic Statistical  Summary
numeric_summary = data.describe()

# Categorical Statistical Summary
categorical_summary = data.describe(include=['object', 'category','bool'])

print("Numberic Statistical  Summary:")
print(numeric_summary)
print("\nCategorical Statistical Summary:")
print(categorical_summary)
print("-----------------------------------------------------------------")  

"""
#pd.set_option('display.max_columns', None)#To show every column
#pd.set_option('display.width', 1000)
"""

plt.figure(figsize=(24,20))

#Histogram for features as Bool and Category
columns=data.columns
index=1
for col in columns:
    if (data[col].dtype=="bool" or data[col].dtype=="category") and col!="Churn" :
        plt.subplot(4, 2, index)
        hist=sns.countplot(x=data[col])  # Seaborn countplot histogram graph
        plt.xlabel(col, fontsize=17)  # Set x-axis label font size
        plt.ylabel('Count', fontsize=17)
        #To put the label to the center of the boxes
        for p in hist.patches:
            hist.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()/2.), 
                        ha='center', va='center', fontsize=25, color='white')
        #plt.title(col)
        index+=1        
        if index>=7:
            plt.figure(figsize=(24,20))
            index=1
            


#for tenure Variable
plt.figure(figsize=(50,20))
hist=sns.countplot(x=data["tenure"])  # Seaborn countplot histogram graph
for p in hist.patches:
    hist.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()/2.), 
                ha='center', va='center', fontsize=12, color='white')
plt.title(col)

#Target description
#-------------------------------------------------------------------------------
plt.figure(figsize=(50,20))
plt.subplot(4, 2, index)
hist=sns.countplot(x=data["Churn"])  # Seaborn countplot histogram graph
#To put the label to the center of the boxes
for p in hist.patches:
    hist.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()/2.), 
                ha='center', va='center', fontsize=12, color='white')
plt.title(col)
plt.ylabel('Customer Count') 

print("Target Data is Churn and Churning Percentages are below")
print()
churn_counts = data['Churn'].value_counts()
churn_percentage = churn_counts / len(data) * 100
print(churn_percentage)

print("-----------------------------------------------------------------------------------")

#Graphs for Numerics
#-----------------------------------------------------------------------------------------
plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = data.boxplot(column='MonthlyCharges',fontsize=15)
fig.set_title('')
fig.set_ylabel('MonthlyCharges',fontsize=15)

plt.subplot(4, 2, 2)
fig = data.boxplot(column='TotalCharges',fontsize=15)
fig.set_title('')
fig.set_ylabel('TotalCharges',fontsize=15)
#------------------------------------------------------------------------------------
#Correlation between numeric values
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for Numerical Features")
plt.show()
print(correlation_matrix)
# ---------------------------------------------------------------------------------------------------
# Feature Engineering Part

df = data_raw.copy()
# data_path = r"C:\Users\sai chandrika\Downloads\Telecom_Churn_Data.xlsx"
# try:
#     df = pd.read_excel(data_path)
#     print("Dataset loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: The file at {data_path} was not found.")
#     exit()
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nOriginal Column Names:")
print(df.columns.tolist())

df.columns = df.columns.str.strip().str.lower()
print("\nColumn Names after standardization:")
print(df.columns.tolist())

object_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nObject-type columns:")
print(object_cols)

for col in object_cols:
    unique_values = df[col].unique()
    print(f"\nUnique values in '{col}':")
    print(unique_values)
    
binary_mappings = {
    'gender': {'Female': 0, 'Male': 1},
    'partner': {'Yes': 1, 'No': 0},
    'dependents': {'Yes': 1, 'No': 0},
    'phoneservice': {'Yes': 1, 'No phone service': 0, 'No': 0},
    'onlinesecurity': {'Yes': 1, 'No internet service': 0, 'No': 0},
    'onlinebackup': {'Yes': 1, 'No': 0},
    'deviceprotection': {'Yes': 1, 'No': 0},
    'techsupport': {'Yes': 1, 'No': 0},
    'streamingtv': {'Yes': 1, 'No': 0},
    'streamingmovies': {'Yes': 1, 'No': 0},
    'paperlessbilling': {'Yes': 1, 'No': 0},
    'churn': {'Yes': 1, 'No': 0}
}


    
for column, mapping in binary_mappings.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)
        print(f"Mapped '{column}' to numerical values.")
    else:
        print(f"Warning: Column '{column}' not found in the dataset.")
        
for col in binary_mappings.keys():
    if col in df.columns:
        unique_values = df[col].unique()
        print(f"\nUnique values in '{col}' after mapping:")
        print(unique_values)
        
print("\nMissing values after mapping:")
print(df.isnull().sum())
        
categorical_columns = ['contract', 'internetservice']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
print("\nColumns after one-hot encoding:")
print(df.columns.tolist())

numeric_conversion_cols = ['totalcharges', 'monthlycharges']
for col in numeric_conversion_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted '{col}' to numeric.")
    else:
        print(f"Warning: Column '{col}' not found in the dataset.")

if 'tenure' in df.columns:
    print("\n'Tenure' column is present.")
else:
    print("\nError: 'tenure' column is missing from the DataFrame.")
    exit()

if 'customerid' in df.columns:
    df.drop('customerid', axis=1, inplace=True)
    print("Dropped 'customerid' column.")
else:
    print("'customerid' column not found in the DataFrame.")

print("\nMissing values before imputation:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("\nNumeric columns identified for imputation:")
print(numeric_cols)

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print("\nFilled missing values in numeric columns with their respective means.")

print("\nMissing values after imputation:")
print(df.isnull().sum())
 
# Creating 'charges_per_month'
df['charges_per_month'] = df['totalcharges'] / df['tenure'].replace(0, 1)
print("\nCreated 'charges_per_month' feature.")

# Creating 'high_monthly_charge'
median_charge = df['monthlycharges'].median()
df['high_monthly_charge'] = (df['monthlycharges'] > median_charge).astype(int)
print("Created 'high_monthly_charge' feature based on median of 'monthlycharges'.")


df_f = df[[
    'gender', 'partner', 'dependents', 'phoneservice', 'onlinesecurity', 
    'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 
    'streamingmovies', 'paperlessbilling', 'churn'
]]

correlation_matrix = df_f.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# output_path = r"C:\Users\sai chandrika\Downloads\Telecom_Churn_Data_Preprocessed.xlsx"
# try:
#     df.to_excel(output_path, index=False)
#     print(f"\nDataFrame successfully saved to {output_path}")
# except Exception as e:
#     print(f"Error saving DataFrame to Excel: {e}")
# -----------------------------------------------------------------------------------------------
# Feature Scaling
# Load the dataset (use your file path)
# data = pd.read_excel(r"C:\Users\Mohammad Arqam\Downloads\Telecom_Churn_Data_Preprocessed.xlsx")
# data_new = data.copy()
data_new = df.copy()

avg_tenure = data_new['tenure'].mean().round()

#correcting data descrepency
round_col = ['onlinebackup','deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']

data_new[round_col] = data_new[round_col].apply(lambda x: x.round())

# Create a binary feature: Is Fiber Optic (1 if Fiber optic, 0 otherwise)
data_new['long_term_customer'] = data_new['tenure'].apply(lambda x: 1 if x > avg_tenure else 0)

#tansposing and applying one-hot encoding in payment_method
data_new = pd.get_dummies(data_new, columns=['paymentmethod'], drop_first=True)

#converting categorical data into 0 and 1
binary_columns = ['paymentmethod_Credit card (automatic)',
'paymentmethod_Electronic check', 'paymentmethod_Mailed check','multiplelines','contract_One year','contract_Two year','internetservice_Fiber optic','internetservice_No']

data_new[binary_columns] = data_new[binary_columns].apply(lambda x: x.map({True: 1, False: 0,'No phone service': 0, 'No': 0,'Yes':1}))

#feature Scaling
# Select the numeric columns for scaling
numeric_columns = ['tenure', 'monthlycharges', 'totalcharges']
scaler = StandardScaler()
# Apply Standardization
standard_scaled = pd.DataFrame(scaler.fit_transform(data_new[numeric_columns]), columns=numeric_columns)

data_standard_scaled = data_new.copy()

for col in numeric_columns:
    data_standard_scaled[col + '_standard_scaled'] = standard_scaled[col]

# Create a figure to hold the plots
plt.figure(figsize=(15, 6))

# Plot original and scaled features
for i, col in enumerate(numeric_columns):
    plt.subplot(1, len(numeric_columns), i + 1)
        
    # Plot original data
    sns.histplot(data_new[col], kde=True, color='blue', label='Original', stat='density', bins=20, alpha=0.5)
    
    # Plot scaled data
    sns.histplot(data_standard_scaled[col + '_standard_scaled'], kde=True, color='orange', label='Scaled', stat='density', bins=20, alpha=0.5)
    
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    
plt.tight_layout()
plt.show()

# Display the first few rows of the scaled data
print("\nStandard Scaled Data:")
print(data_standard_scaled.head())

#dropping columns after scaling
columns_to_drop = ['charges_per_month','tenure', 'monthlycharges', 'totalcharges']

data_standard_scaled.drop(columns=columns_to_drop,inplace = True)

# data_standard_scaled.to_csv('Telecom_Churn_Standard_Scaled.csv', index=False)

#print(data.drop(columns=[SeniorCitizen]).describe().T)
print("-----------------------------------------------------------------------------------")
#SVM Prediction Model
#------------------------
#file_path = os.path.dirname(os.path.abspath(__file__))
#file_path = file_path.replace("\\", "/")
# Load the dataset from a CSV file
data = data_standard_scaled.copy()
# Define features (X) and target (y)
X = data.drop('churn', axis=1)
y = data['churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM with Linear kernel
linear_model = svm.SVC(kernel='linear', random_state=42)
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear, average='macro')
precision_linear = precision_score(y_test, y_pred_linear, average='macro')
f1_linear = f1_score(y_test, y_pred_linear, average='macro')
classification_report_linear = classification_report(y_test, y_pred_linear)

print("Results for Linear Kernel:")
print(f"Accuracy: {round(accuracy_linear, 2)}")
print(f"Recall: {round(recall_linear, 2)}")
print(f"Precision: {round(precision_linear, 2)}")
print(f"F1 Score: {round(f1_linear, 2)}")
print(classification_report_linear)
print("\n" + "-"*50 + "\n")

# SVM with Polynomial kernel
poly_model = svm.SVC(kernel='poly', random_state=42)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
recall_poly = recall_score(y_test, y_pred_poly, average='macro')
precision_poly = precision_score(y_test, y_pred_poly, average='macro')
f1_poly = f1_score(y_test, y_pred_poly, average='macro')
classification_report_poly = classification_report(y_test, y_pred_poly)

print("Results for Polynomial Kernel:")
print(f"Accuracy: {round(accuracy_poly, 2)}")
print(f"Recall: {round(recall_poly, 2)}")
print(f"Precision: {round(precision_poly, 2)}")
print(f"F1 Score: {round(f1_poly, 2)}")
print(classification_report_poly)
print("\n" + "-"*50 + "\n")

# SVM with RBF kernel
rbf_model = svm.SVC(kernel='rbf', random_state=42)
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf, average='macro')
precision_rbf = precision_score(y_test, y_pred_rbf, average='macro')
f1_rbf = f1_score(y_test, y_pred_rbf, average='macro')
classification_report_rbf = classification_report(y_test, y_pred_rbf)

print("Results for RBF Kernel:")
print(f"Accuracy: {round(accuracy_rbf, 2)}")
print(f"Recall: {round(recall_rbf, 2)}")
print(f"Precision: {round(precision_rbf, 2)}")
print(f"F1 Score: {round(f1_rbf, 2)}")
print(classification_report_rbf)
print("\n" + "-"*50 + "\n")

# SVM with Sigmoid kernel
sigmoid_model = svm.SVC(kernel='sigmoid', random_state=42)
sigmoid_model.fit(X_train, y_train)
y_pred_sigmoid = sigmoid_model.predict(X_test)
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
recall_sigmoid = recall_score(y_test, y_pred_sigmoid, average='macro')
precision_sigmoid = precision_score(y_test, y_pred_sigmoid, average='macro')
f1_sigmoid = f1_score(y_test, y_pred_sigmoid, average='macro')
classification_report_sigmoid = classification_report(y_test, y_pred_sigmoid)

print("Results for Sigmoid Kernel:")
print(f"Accuracy: {round(accuracy_sigmoid, 2)}")
print(f"Recall: {round(recall_sigmoid, 2)}")
print(f"Precision: {round(precision_sigmoid, 2)}")
print(f"F1 Score: {round(f1_sigmoid, 2)}")
print(classification_report_sigmoid)


print("-----------------------------------------------------------------------------")

print("----------------------------------DECISION TREE------------------------------")


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
# data = pd.read_csv(r"C:\Users\Mohammad Arqam\Telecom_Churn_Standard_Scaled.csv")
X = data.drop('churn', axis=1)
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entropy
print('Model with Entropy Parameter')
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))


#generate and display the confusion matrix after pruning
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',linewidths=.5,square=True,cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('predict label')
all_sample_title = 'Accuaracy: {0}'.format(accuracy)
plt.title(all_sample_title,size=15)
plt.show()

#vizualise
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns,
               class_names=['Class1','Class2'])
plt.show()


#Gini
print('Model with Gini Parameter')
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))


#generate and display the confusion matrix after pruning
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',linewidths=.5,square=True,cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('predict label')
all_sample_title = 'Accuaracy: {0}'.format(accuracy)
plt.title(all_sample_title,size=15)
plt.show()

#vizualise
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns,
               class_names=['Class1','Class2'])
plt.show()

print('Optimization Technique using Grid Search')
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth':[None, 10, 20, 30, 40],
    'min_samples_split':[2, 10, 20],
    'min_samples_leaf':[1, 5, 10],
    'max_features':[None,'sqrt','log2'],
}
grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1,verbose=1)

grid_search.fit(X_train,y_train)

best_model = grid_search.best_estimator_

y_pred=best_model.predict(X_test)
precision=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)
accuray = accuracy_score(y_test,y_pred)

print(f"Best parameters:{grid_search.best_params_}")
print(f"precision:{precision}")
print(f"F1 score: {f1}")
print(f"Accuray: {accuray}")
print("confusion matrix:")
print(conf_matrix)
print("classification Report:")
print(classification_report(y_test,y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.title('confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')
plt.show()


#Pruning
print('Model after pruning')
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

#initializing variable to find the best alpha
best_alpha = 0
best_accuracy = 0

#Iterate over different vlaues of ccp_alpha to find the best score
for ccp_alpha in ccp_alphas:
    model.set_params(ccp_alpha=ccp_alpha)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy>best_accuracy:
        best_alpha = ccp_alpha
        best_accuracy = accuracy
        
model.set_params(ccp_alpha=best_alpha)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))

#generate and display the confusion matrix after pruning
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',linewidths=.5,square=True,cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('predict label')
all_sample_title = 'Accuaracy: {0}'.format(accuracy)
plt.title(all_sample_title,size=15)
plt.show()

#vizualise
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns,
               class_names=['Class1','Class2'])
plt.show()

print('-----------------------------------------------------------------------------')
print('----------------------------RANDOM FOREST------------------------------------')
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Load the dataset
# data = pd.read_csv(r"C:\Users\sai chandrika\Downloads\Telecom_Churn_Standard_Scaled.csv")
# Check for missing values and handle them by filling with the mean
if data.isnull().sum().sum() > 0:
    print("Missing values detected. Filling with column mean.")
    data = data.fillna(data.mean())

# Separate features and target
X = data.drop(columns='churn')
y = data['churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=10,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print('Initial Model Performance:')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [5, 10, 15],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize GridSearchCV with the RandomForest classifier and the parameter grid
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print(f'Best parameters: {grid_search.best_params_}')

# Evaluate the best model on the testing data
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best, average='weighted')
best_recall = recall_score(y_test, y_pred_best, average='weighted')
best_f1 = f1_score(y_test, y_pred_best, average='weighted')
best_conf_matrix = confusion_matrix(y_test, y_pred_best)

print('\nBest Model Performance After Hyperparameter Tuning:')
print(f'Accuracy: {best_accuracy * 100:.2f}%')
print(f'Precision: {best_precision * 100:.2f}%')
print(f'Recall: {best_recall * 100:.2f}%')
print(f'F1 Score: {best_f1 * 100:.2f}%')
print('Confusion Matrix:')
print(best_conf_matrix)
