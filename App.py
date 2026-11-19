import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

# Data Load
df=pd.read_csv(r"C:\Users\Amreen\Documents\EDA_PROJECT\Loan_Prediction_Train.csv")

#--------------------------------------------------------------------


# st.subheader("Step-1 : Missing Values")
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
# st.write(df.isnull().sum())

# #--------------missing values fill---------
# 1. Pehle number me convert karo (bahut zaruri)
df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')
df['Loan_Amount_Term'] = pd.to_numeric(df['Loan_Amount_Term'], errors='coerce')
df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce')
df['Loan_Status'] = pd.to_numeric(df['Loan_Status'], errors='coerce')

# # 2. Fill Missing values 
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)  # direct 360 bhar do, mode ka chakkar khatam
df['Credit_History'] = df['Credit_History'].fillna(1.0) # direct 1 bhar do

st.title("EDA-Exploratory Data Analysis")
st.subheader("Dataset of 614 Bank Loan Applicants")
st.write("_____________________________________________________________________________________________")

#-------------filter property Area wise-------
st.subheader('Applicants Property Area Wise')
area=st.selectbox("Select Area",df['Property_Area'].unique())
st.write(df[df['Property_Area']==area])

# df.describe()
st.write("_____________________________________________________________________________________________")
st.subheader('Describe Method')
st.write('This is use to calculate (count,min,max,25%,50%,75%,mean, and standard deviation) of Dataset')
df_des=df.describe()
st.write(df_des)
st.write("_____________________________________________________________________________________________")


# pie chart

keys = df['Education'].value_counts().index
val = df['Education'].value_counts().values
fig, ax = plt.subplots(figsize=(3, 2))
ax.pie(
    val,
    labels=keys,
    startangle=45,
    autopct='%1.2f%%',
    explode=[0.1, 0.1],
    radius=1.4
)
st.subheader("Pie Chart : Graduate vs Non Graduate")
st.pyplot(fig)
plt.close(fig)
st.write("_____________________________________________________________________________________________")

# Male and Female - Self Employed Yes/No
male_yes = len(df[(df["Gender"] == "Male") & (df["Self_Employed"] == "Yes")])
male_no = len(df[(df["Gender"] == "Male") & (df["Self_Employed"] == "No")])

female_yes = len(df[(df["Gender"] == "Female") & (df["Self_Employed"] == "Yes")])
female_no = len(df[(df["Gender"] == "Female") & (df["Self_Employed"] == "No")])

fig, ax = plt.subplots(figsize=(5,3))

x = np.arange(2)
width = 0.25

ax.bar(x - width/2, [male_yes, female_yes], width,
       label="Self Employed: Yes", color="red")

ax.bar(x + width/2, [male_no, female_no], width,
       label="Self Employed: No", color="green")

ax.set_xticks(x)
ax.set_xticklabels(["Male", "Female"])
ax.set_xlabel("Gender")
ax.set_ylabel("Number of Applicants")
st.subheader("Histogram : Gender vs Self Employed")
ax.legend()

st.pyplot(fig)
plt.close(fig)
st.write("_____________________________________________________________________________________________")

#--------------Encoding-------------------
# from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
    df[col] = df[col].fillna(df[col].mode()[0])
# tab1, tab2 = st.tabs(["Level 1 & 2: Project Story", "Live Predictor"])
#--------------------------------------------------------------------

# Skewness

st.subheader(" Skewness: Income Right-Skewed")
skew = df['ApplicantIncome'].skew()
st.write(f"Skewness = {skew:.2f}")
st.write("90% less income applicants, 10% high income applicants. Thats why apply transfom")
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.histplot(df['ApplicantIncome'], kde=True, ax=ax[0])
ax[0].set_title("Original Skewed")
sns.histplot(np.log1p(df['ApplicantIncome']), kde=True, ax=ax[1])
ax[1].set_title("Normal after Transformed")
st.pyplot(fig)
st.write("_____________________________________________________________________________________________")

# Scatter PLot

st.subheader("Scatter Plot: Applicant Income vs Loan Amount")
st.write("Credit History Essential for Loan Approval or Rejection")
st.write("Applicant's Credit History help to know previous of loans")
st.write(" 1 = Good , 0 = Bad")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="ApplicantIncome",
    y="LoanAmount",
    hue="Credit_History",
    palette={1: "green", 0: "red"},
    s=50,
    ax=ax)

ax.set_title("Applicant Income vs Loan Amount")
ax.set_xlabel("Applicant Income")
ax.set_ylabel("Loan Amount")

ax.legend(title="Credit_History")
st.pyplot(fig)
plt.close(fig)
st.write("_____________________________________________________________________________________________")

# Boxplot

st.subheader("Box Plot: Education vs Applicant Income")

fig, ax = plt.subplots(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="Education",
    y="ApplicantIncome",
    ax=ax
)

ax.set_title("Education vs Applicant Income")
ax.set_xlabel("Education")
ax.set_ylabel("Applicant Income")

st.pyplot(fig)
plt.close(fig)
st.write("_____________________________________________________________________________________________")

                
# Correlation
st.subheader(" Correlation Heatmap: Which is Outlier ?") 
corr = df[['ApplicantIncome','LoanAmount','Credit_History','Loan_Status']].corr()
fig2, ax2 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)
st.success("Insight: Loan_Status of Credit_History and correlation = 0.79 , important feature.")
st.write("_____________________________________________________________________________________________")

            #-----standardization----------
st.subheader("Scaling Mehtods")
st.subheader("1.Standardization Method  – StandardScaler")
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

df['ApplicantIncome_Standard'] = ss.fit_transform(
    df[['ApplicantIncome']]
)
st.write(df[['ApplicantIncome', 'ApplicantIncome_Standard']].head())

# normalization
st.subheader("2.Normalization Method  – MinMaxScaler")
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df['ApplicantIncome_Normalized'] = ms.fit_transform(
    df[['ApplicantIncome']]
)

st.write(df[['ApplicantIncome', 'ApplicantIncome_Normalized']].head())

#Compare Both Methods
st.subheader("Compare Both Methods")

st.write(df[[
    'ApplicantIncome',
    'ApplicantIncome_Standard',
    'ApplicantIncome_Normalized'
]].head(10))

#Before Scaling
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df['ApplicantIncome'], kde=True, ax=ax)
ax.set_title("Before Scaling - Applicant Income")
st.pyplot(fig)
plt.close(fig)

#--------------------------

#"Box Plot - Scaling Comparison"
st.subheader("Box Plot - Scaling Comparison")
fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
sns.boxplot(
    data=df[[
        'ApplicantIncome_Standard',
        'ApplicantIncome_Normalized'
    ]],
    orient='h',
    ax=ax)
ax.set_title("Standardization vs Normalization")
st.pyplot(fig)
plt.close(fig)