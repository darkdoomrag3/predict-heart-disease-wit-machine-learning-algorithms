from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import statistics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# %matplotlib notebook
from google.colab import drive
drive.mount('/content/gdrive')
df = pd.read_csv('/content/gdrive/MyDrive/heart.csv')
df.head()

# data attribute information
# Attribute Information:

# 1.age 2.sex (1= Male, 0= Female) 3.chest pain type (4 values) 4.resting blood pressure 5.serum cholestoral in mg/dl 6.fasting blood sugar > 120 mg/dl 7.resting electrocardiographic results (values 0,1,2) 8.maximum heart rate achieved 9.exercise induced angina 10.oldpeak = ST depression induced by exercise relative to rest 11.the slope of the peak exercise ST segment 12.number of major vessels (0-3) colored by flourosopy 13.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect


df.tail()
df.shape
df.info()
df.describe().T
df.oldpeak.unique()
df.isna().sum()

# There is no null value present.
# Let's find how many people have heart disease and how many people doesn't have heart disease
df['sex'].value_counts()
df['sex'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
df['target'].value_counts()
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])

labels = 'Male', 'Female'
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(df.sex.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
fig1.savefig("/content/gdrive/MyDrive/...")
plt.show()

labels = "Has heart disease", "Doesn't have heart disease"
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(df.target.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
fig1.savefig("/content/gdrive/MyDrive/...")
plt.show()
fig = sns.countplot(x='target', data=df, hue='sex')
fig.set_xticklabels(
    labels=["Doesn't have heart disease", 'Has heart disease'], rotation=0)
plt.legend(['Female', 'Male'])
fig.savefig("/content/gdrive/MyDrive/")
plt.title("Heart Disease Frequency for Sex")
df.cp.value_counts()
# plotting a bar chart
fig1 = df.cp.value_counts().plot(
    kind='bar', color=['salmon', 'lightskyblue', 'springgreen', 'khaki'])
fig1.set_xticklabels(
    labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)
fig1.figure.savefig("/content/gdrive/MyDrive/...")
plt.title('Chest pain type vs count')

pd.crosstab(df.sex, df.cp)
fig = pd.crosstab(df.sex, df.cp).plot(kind='bar', color=[
    'coral', 'lightskyblue', 'plum', 'khaki'])
plt.title('Type of chest pain for sex')
fig.set_xticklabels(labels=['Female', 'Male'], rotation=0)
fig.figure.savefig("/content/gdrive/MyDrive/")
plt.legend(['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'])

pd.crosstab(df.cp, df.target)
plt.figure(figsize=(14, 6))
fig = sns.countplot(x='cp', data=df, hue='target')
fig.set_xticklabels(
    labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)
plt.legend(['No disease', 'disease'])


# create a distribution plot with normal distribution curve
sns.displot(x='age', data=df, bins=30, kde=True)
skewness = str(df["age"].skew())
kurtosis = str(df["age"].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.savefig("/content/gdrive/MyDrive/")
plt.show()

sns.displot(x='thalach', data=df, bins=30, kde=True, color='chocolate')
skewness = str(df["thalach"].skew())
kurtosis = str(df["thalach"].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.savefig("/content/gdrive/MyDrive/")
plt.show()

# Creating a figure
plt.figure(figsize=(10, 6))

# plotting the values for people who have heart disease
plt.scatter(df.age[df.target == 1],
            df.thalach[df.target == 1],
            c="tomato")

# plotting the values for people who doesn't have heart disease
plt.scatter(df.age[df.target == 0],
            df.thalach[df.target == 0],
            c="lightgreen")

# Addind info
plt.title("Heart Disease w.r.t Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate")
plt.savefig("/content/gdrive/MyDrive/")

sns.kdeplot(x='age', y='thalach', data=df, color='darkcyan')


sns.displot(x=df.thalach[df.target == 1], data=df, kde=True, color='olive')
skewness = str(df.thalach[df.target == 1].skew())
kurtosis = str(df.thalach[df.target == 1].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.title("Maximum heart achieved of peple with heart disease")
plt.xlabel("Maximum heart rate achieved")
plt.ylabel("Number of people with heart disease")


sns.displot(x=df.thalach[df.target == 0], data=df, kde=True, color='slategray')
skewness = str(df.thalach[df.target == 0].skew())
kurtosis = str(df.thalach[df.target == 0].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.title("Maximum heart achieved of people without heart disease")
plt.xlabel("Maximum heart rate achieved")
plt.ylabel("Number of people without heart disease")

sns.displot(x='chol', data=df, bins=30, kde=True, color='teal')
skewness = str(df['chol'].skew())
kurtosis = str(df['chol'].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))


# Creating another figure
plt.figure(figsize=(10, 6))

# plotting the values for people who have heart disease
plt.scatter(df.age[df.target == 1],
            df.chol[df.target == 1],
            c="salmon")  # define it as a scatter figure

# plotting the values for people who doesn't have heart disease
plt.scatter(df.age[df.target == 0],
            df.chol[df.target == 0],
            c="lightblue")  # axis always come as (x, y)

# Add some helpful info
plt.title("Heart Disease w.r.t Age and Serum Cholestoral")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Serum cholestoral")
plt.savefig("/content/gdrive/MyDrive/")

sns.kdeplot(x='age', y='chol', data=df, color='firebrick')


sns.displot(x=df.chol[df.target == 1], data=df, kde=True, color='dodgerblue')
skewness = str(df.chol[df.target == 1].skew())
kurtosis = str(df.chol[df.target == 1].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.title("Serum Cholestoralof people with heart disease")
plt.xlabel("Serum Cholestoral")
plt.ylabel("Number of people with heart disease")


sns.displot(x=df.chol[df.target == 0], data=df, kde=True, color='forestgreen')
skewness = str(df.chol[df.target == 0].skew())
kurtosis = str(df.chol[df.target == 0].kurt())
plt.legend([skewness, kurtosis], title=("skewness and kurtosis"))
plt.title("Serum Cholestoralof people without heart disease")
plt.xlabel("Serum Cholestoral")
plt.ylabel("Number of people without heart disease")


pd.crosstab(df.exang, df.sex)


fig = sns.countplot(x='exang', data=df, hue='sex')
plt.title('Exercise induced angina for sex')
fig.set_xticklabels(labels=["Doesn't have exang", 'Has exang'], rotation=0)
plt.legend(['Female', 'Male'])
df.fbs.value_counts()

# visualizing in Pie chart
labels = 'fbs<120 mg/dl', 'fbs>120 mg/dl'
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(df.fbs.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
pd.crosstab(df.sex, df.fbs)

fig = pd.crosstab(df.sex, df.fbs).plot(
    kind='bar', color=['lightblue', 'salmon'])
plt.title("Fasting blood sugar w.r.t sex")
fig.set_xticklabels(labels=['fbs>120 mg/dl', 'fbs<120 mg/dl'], rotation=0)
plt.legend(['Female', 'Male'])


corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

cat_values = []
conti_values = []

for col in df.columns:
    if len(df[col].unique()) >= 10:
        conti_values.append(col)
    else:
        cat_values.append(col)

print("catageroy values: ", cat_values)
print("continous values: ", conti_values)

plt.figure(figsize=(18, 8))

for i, col in enumerate(conti_values, 1):
    plt.subplot(2, 3, i)
    df[df.target == 1][col].hist(
        bins=40, color='red', alpha=0.5,  label='Disease: YES')
    df[df.target == 0][col].hist(
        bins=40, color='blue', alpha=0.5,  label='Disease: NO')
    plt.xlabel(col)
    plt.legend()

#
# trestbps[resting bp] anything above 130-140 is generally of concern
# chol[cholesterol] greater than 200 is of concern
# thalach People over 140 value are more likely to have heart disease
# oldpeak with value 0 are more than likely to have heart disease than any other value

# checking for null values
df.isna().sum()

# Checking Correlation using Heatmap
x = df.corr()
plt.figure(figsize=(18, 8))
sns.heatmap(x, annot=True)
plt.savefig("/content/gdrive/MyDrive/")

df.describe().T


x = 1
plt.figure(figsize=(20, 20))

for i in df.columns:
    plt.subplot(4, 4, x)
    plt.boxplot(df[i])
    plt.title(i)
    x = x+1


q1 = df['trestbps'].quantile(q=0.25)
q3 = df["trestbps"].quantile(q=0.75)
IQR = q3 - q1

IQR_lower_limit = int(q1 - (1.5*IQR))
IQR_upper_limit = int(q3 + (1.5*IQR))

print("Upper limit of IQR:", IQR_upper_limit)
print("Lower limit of IQR:", IQR_lower_limit)

cleaned_data = df[df["trestbps"] < IQR_upper_limit]

plt.boxplot(cleaned_data["trestbps"])


cat_values.remove('target')
cleaned_data = pd.get_dummies(cleaned_data, columns=cat_values)

X = cleaned_data.drop(columns='target')
y = cleaned_data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1)

sc = StandardScaler()
X_train[conti_values] = sc.fit_transform(X_train[conti_values])
X_test[conti_values] = sc.transform(X_test[conti_values])

logreg = LogisticRegression()
lr = LogisticRegression()
logreg.fit(X_train, y_train)
lr.fit(X_train, y_train)
lr_acc_score = accuracy_score(y_test, y_pred_test)
lr_acc_score

confusion_matrix(y_test, y_pred_test)

accuracies = {}
roc_score = roc_auc_score(y_test, y_pred_test)
lr = LogisticRegression()
accuracies['Logistic Regression'] = roc_score
roc_score
plt.plot(tpr, fpr, color='blue', label='ROC')
plt.plot([0, 1], [0, 1], color='black',
         label='ROC curve (area = %0.2f)' % roc_score)
plt.xlabel("False Positivity Rate")
plt.ylabel("True Positivity Rate")
plt.title("Reciever Operator Characteristic curve")
plt.legend()
plt.show()


m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train, y_train)
nbpred = nb.predict(X_test)
nb_acc_score = accuracy_score(y_test, nbpred)
accuracies['Naive Bayes'] = nb_acc_score
print(nb_acc_score)

m3 = 'Random Forest Classfier'

rf = RandomForestClassifier(n_estimators=20, random_state=12, max_depth=5)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)
rf_acc_score = accuracy_score(y_test, rf_predicted)
accuracies['Random Forest'] = rf_acc_score
print(rf_acc_score)
m4 = 'K-Neighbors Classifier'

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_acc_score = accuracy_score(y_test, knn_predicted)
accuracies['KNN'] = knn_acc_score
print(knn_acc_score)

m5 = 'Decision Tree Classifier'

dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_acc_score = accuracy_score(y_test, dt_predicted)
accuracies['Decision Tree'] = dt_acc_score
print(dt_acc_score)

svm = SVC(random_state=1)

m6 = 'SVM'

svm = SVC(random_state=1)
svm.fit(X_train, y_train)
svm_acc_score = svm.score(X_test, y_test)
accuracies['SVM'] = svm_acc_score
print(svm_acc_score)

model_ev = pd.DataFrame({'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest',
                                   'K-Nearest Neighbour', 'Decision Tree'], 'Accuracy': [lr_acc_score*100,
                                                                                         nb_acc_score*100, rf_acc_score*100, knn_acc_score*100, dt_acc_score*100]})
model_ev
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

x.head()
y.head()
# spltting the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=31)
len(x_train), len(x_test), len(y_train), len(y_test)
x_train.head()
y_train.head()

m = df.groupby('target').mean()
s = df.groupby('target').std()
confidence = 0.95
supp = df.groupby('target').mean()
conf = df.groupby('target').confi()
rule = apriori(transactions=transacts, min_support=sup, min_confidence=confidance, min_lift=gcb_grid(
    'min_samples_split'), min_length=gcb_grid('max_depth'), max_length=gcb_grid('min_samples_leaf'))
gbc_grid = {'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200, 500, 1000],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_depth': [1, 2, 3]}


gbc_gscv = GridSearchCV(GradientBoostingClassifier(),
                        param_grid=gbc_grid,
                        cv=5,
                        verbose=True)
gbc_gscv.fit(x_train, y_train)
gbc_tuned_score = gbc_gscv.score(x_test, y_test)
gbc_tuned_score
log_grid = {'C': np.logspace(-5, 5),
            'solver': ['liblinear'],
            'max_iter': np.arange(1000, 2000, 100),
            'penalty': ['l1', 'l2']
            }

log_gscv = GridSearchCV(LogisticRegression(random_state=7),
                        param_grid=log_grid,
                        cv=5,
                        verbose=True)

log_gscv.fit(x_train, y_train)
log_tuned_score = log_gscv.score(x_test, y_test)
log_tuned_score

colors = ["purple", "green", "orange", "magenta", "#CFC60E", "#0FBBAE", "red"]

sns.set_style("whitegrid")
plt.figure(figsize=(20, 5))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(
    accuracies.values()), palette=colors)
plt.show()
plt.savefig("/content/gdrive/MyDrive", dpi=300)


m5 = 'Decision Tree Classifier'

dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_acc_score = accuracy_score(y_test, dt_predicted)
accuracies['Decision Tree'] = dt_acc_score
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
print(dt_acc_score)
print("confussion matrix")
print(dt_conf_matrix)

sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, dt_predicted):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(
        y_test, dt_predicted), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Decision Tree")
    plt.show()


plot_conf_matrix(y_test, dt_predicted)
plt.savefig("/content/gdrive/MyDrive/")

m3 = 'Random Forest Classfier'

rf = RandomForestClassifier(n_estimators=20, random_state=12, max_depth=5)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)
rf_acc_score = accuracy_score(y_test, rf_predicted)
accuracies['Random Forest'] = rf_acc_score
print(rf_acc_score)

rf_conf_matrix = confusion_matrix(y_test, rf_predicted)


sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, rf_predicted):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(
        y_test, rf_predicted), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.title("Random Forest")
    plt.ylabel("True Labels")
    plt.show()


plot_conf_matrix(y_test, rf_predicted)
plt.savefig("/content/gdrive/MyDrive/")


m4 = 'K-Neighbors Classifier'

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_acc_score = accuracy_score(y_test, knn_predicted)
accuracies['KNN'] = knn_acc_score
print(knn_acc_score)

rf_conf_matrix = confusion_matrix(y_test, knn_predicted)
sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, knn_predicted):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(
        y_test, knn_predicted), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.title("KNN")
    plt.ylabel("True Labels")
    plt.show()


plot_conf_matrix(y_test, knn_predicted)
plt.savefig("/content/gdrive/MyDrive/")

svm = SVC(random_state=1)

m6 = 'SVM'

svm = SVC(random_state=1)
svm.fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
svm_acc_score = svm.score(X_test, y_test)

accuracies['SVM'] = svm_acc_score
print(svm_acc_score)

rf_conf_matrix = confusion_matrix(y_test, svm_predicted)
sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, svm_predicted):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(
        y_test, svm_predicted), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.title("SVM")
    plt.ylabel("True Labels")
    plt.show()


plot_conf_matrix(y_test, svm_predicted)
plt.savefig("/content/gdrive/MyDrive/")


m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train, y_train)
nbpred = nb.predict(X_test)
nb_acc_score = accuracy_score(y_test, nbpred)
accuracies['Naive Bayes'] = nb_acc_score
print(nb_acc_score)

nb_conf_matrix = confusion_matrix(y_test, nbpred)
sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, nbpred):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(y_test, nbpred), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.title("Naive Bayes")
    plt.ylabel("True Labels")
    plt.show()


plot_conf_matrix(y_test, nbpred)
plt.savefig("/content/gdrive/MyDrive/")


confusion_matrix(y_test, y_pred_test)
sns.set(font_scale=1.5)


def plot_conf_matrix(y_test, y_pred_test):
    """
    Plots a nice looking confusion matrix using Seaborns heatmap
    """

    fig, ax = plt.subplots(figsize=(8, 5), dpi=90)
    ax = sns.heatmap(confusion_matrix(
        y_test, y_pred_test), annot=True, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.title("Logistic Regression")
    plt.ylabel("True Labels")
    plt.show()


plot_conf_matrix(y_test, y_pred_test)
plt.savefig("/content/gdrive/MyDrive/")


model_ev = pd.DataFrame({'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest',
                                   'K-Nearest Neighbour', 'Decision Tree'], 'Accuracy': [lr_acc_score*100,
                                                                                         nb_acc_score*100, rf_acc_score*100, knn_acc_score*100, dt_acc_score*100]})
model_ev
