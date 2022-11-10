import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#%matplotlib notebook
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

#There is no null value present.
#Let's find how many people have heart disease and how many people doesn't have heart disease
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
fig = sns.countplot(x = 'target', data = df, hue = 'sex')
fig.set_xticklabels(labels=["Doesn't have heart disease", 'Has heart disease'], rotation=0)
plt.legend(['Female', 'Male'])
fig.savefig("/content/gdrive/MyDrive/")
plt.title("Heart Disease Frequency for Sex")
df.cp.value_counts()
#plotting a bar chart
fig1 = df.cp.value_counts().plot(kind = 'bar', color = ['salmon', 'lightskyblue', 'springgreen', 'khaki'])
fig1.set_xticklabels(labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)
fig1.figure.savefig("/content/gdrive/MyDrive/...")
plt.title('Chest pain type vs count')

pd.crosstab(df.sex, df.cp)
fig = pd.crosstab(df.sex, df.cp).plot(kind = 'bar', color = ['coral', 'lightskyblue', 'plum', 'khaki'])
plt.title('Type of chest pain for sex')
fig.set_xticklabels(labels=['Female', 'Male'], rotation=0)
fig.figure.savefig("/content/gdrive/MyDrive/")
plt.legend(['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'])

pd.crosstab(df.cp, df.target)
plt.figure(figsize=(14,6))
fig = sns.countplot(x = 'cp', data = df, hue = 'target')
fig.set_xticklabels(labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)
plt.legend(['No disease', 'disease'])


#create a distribution plot with normal distribution curve
sns.displot( x = 'age', data = df, bins = 30, kde = True)
skewness=str(df["age"].skew())
kurtosis=str(df["age"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.savefig("/content/gdrive/MyDrive/")
plt.show()

sns.displot(x = 'thalach', data = df, bins = 30, kde = True, color = 'chocolate')
skewness=str(df["thalach"].skew())
kurtosis=str(df["thalach"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.savefig("/content/gdrive/MyDrive/")
plt.show()

# Creating a figure
plt.figure(figsize=(10,6))

#plotting the values for people who have heart disease
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="tomato")

#plotting the values for people who doesn't have heart disease
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightgreen")

# Addind info
plt.title("Heart Disease w.r.t Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate")
plt.savefig("/content/gdrive/MyDrive/")

sns.kdeplot(x = 'age', y = 'thalach', data = df, color = 'darkcyan')


sns.displot(x = df.thalach[df.target==1], data = df, kde = True, color= 'olive')
skewness=str(df.thalach[df.target==1].skew())
kurtosis=str(df.thalach[df.target==1].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("Maximum heart achieved of peple with heart disease")
plt.xlabel("Maximum heart rate achieved")
plt.ylabel("Number of people with heart disease")


sns.displot(x = df.thalach[df.target==0], data = df, kde = True, color= 'slategray')
skewness=str(df.thalach[df.target==0].skew())
kurtosis=str(df.thalach[df.target==0].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("Maximum heart achieved of people without heart disease")
plt.xlabel("Maximum heart rate achieved")
plt.ylabel("Number of people without heart disease")

sns.displot(x = 'chol', data = df, bins = 30, kde = True, color = 'teal')
skewness=str(df['chol'].skew())
kurtosis=str(df['chol'].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))


# Creating another figure
plt.figure(figsize=(10,6))

#plotting the values for people who have heart disease
plt.scatter(df.age[df.target==1], 
            df.chol[df.target==1], 
            c="salmon") # define it as a scatter figure

#plotting the values for people who doesn't have heart disease
plt.scatter(df.age[df.target==0], 
            df.chol[df.target==0], 
            c="lightblue") # axis always come as (x, y)

# Add some helpful info
plt.title("Heart Disease w.r.t Age and Serum Cholestoral")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Serum cholestoral")
plt.savefig("/content/gdrive/MyDrive/payan name/cholestrol.jpg")

sns.kdeplot(x = 'age', y = 'chol', data = df, color = 'firebrick')


sns.displot(x = df.chol[df.target==1], data = df, kde = True, color= 'dodgerblue')
skewness=str(df.chol[df.target==1].skew())
kurtosis=str(df.chol[df.target==1].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"));
plt.title("Serum Cholestoralof people with heart disease")
plt.xlabel("Serum Cholestoral")
plt.ylabel("Number of people with heart disease")


sns.displot(x = df.chol[df.target==0], data = df, kde = True, color= 'forestgreen')
skewness=str(df.chol[df.target==0].skew())
kurtosis=str(df.chol[df.target==0].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("Serum Cholestoralof people without heart disease")
plt.xlabel("Serum Cholestoral")
plt.ylabel("Number of people without heart disease")


pd.crosstab(df.exang, df.sex)



fig = sns.countplot(x = 'exang', data = df, hue = 'sex')
plt.title('Exercise induced angina for sex')
fig.set_xticklabels(labels=["Doesn't have exang", 'Has exang'], rotation=0)
plt.legend(['Female', 'Male'])
df.fbs.value_counts()

#visualizing in Pie chart
labels = 'fbs<120 mg/dl', 'fbs>120 mg/dl'
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(df.fbs.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
pd.crosstab(df.sex, df.fbs)

fig = pd.crosstab(df.sex, df.fbs).plot(kind = 'bar', color = ['lightblue', 'salmon'])
plt.title("Fasting blood sugar w.r.t sex")
fig.set_xticklabels(labels=['fbs>120 mg/dl', 'fbs<120 mg/dl'], rotation=0)
plt.legend(['Female', 'Male'])


