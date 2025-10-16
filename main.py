# main.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="🚢 Titanic Survival Prediction", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📚 Navigation")
menu = st.sidebar.radio(
    "Go to Section:",
    ["🏠 Overview", "📊 Data Exploration", "🤖 Model Training & Evaluation", "🧍 Passenger Survival Prediction"]
)

# --- Load Data ---
@st.cache_data
def load_data():
    data = pd.read_csv('./train.csv')
    data = data.drop(columns='Cabin', axis=1)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data.replace({'Sex': {'male': 0, 'female': 1},
                  'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
    return data

titanic_data = load_data()

# --- Prepare Data ---
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# --- Train Model ---
model = LogisticRegression()
model.fit(X_train, Y_train)
X_test_Prediction = model.predict(X_test)

# --- Metrics ---
acc = accuracy_score(Y_test, X_test_Prediction)
precision = precision_score(Y_test, X_test_Prediction)
recall = recall_score(Y_test, X_test_Prediction)
f1 = f1_score(Y_test, X_test_Prediction)

# ===============================================================
# 🏠 Overview
# ===============================================================
if menu == "🏠 Overview":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("""
    Welcome to the **Titanic Survival Prediction App** built with Streamlit and Scikit-learn.  
    This app allows you to:
    - Explore the Titanic dataset  
    - Train and evaluate a logistic regression model  
    - Predict whether a passenger would survive based on input details  
    
    Use the sidebar to navigate between sections.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.markdown("---")
    st.write("Developed with ❤️ using **Streamlit** and **Scikit-learn**")

# ===============================================================
# 📊 Data Exploration
# ===============================================================
elif menu == "📊 Data Exploration":
    st.title("📊 Data Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(titanic_data.head())

    st.subheader("Basic Information")
    st.write(titanic_data.describe())

    st.subheader("Missing Values")
    st.write(titanic_data.isnull().sum())

    st.markdown("---")
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Count of Survivors")
        surv_counts = titanic_data['Survived'].value_counts().rename({0: 'Not Survived', 1: 'Survived'})
        st.bar_chart(surv_counts)

    with col2:
        st.write("### Survival Count by Gender")
        gender_survival = titanic_data.groupby('Sex')['Survived'].value_counts().unstack().fillna(0)
        gender_survival.index = ['Male', 'Female']
        st.bar_chart(gender_survival)

    st.write("### Survival Count by Passenger Class")
    pclass_survival = titanic_data.groupby('Pclass')['Survived'].value_counts().unstack().fillna(0)
    st.bar_chart(pclass_survival)

# ===============================================================
# 🤖 Model Training & Evaluation
# ===============================================================
elif menu == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    st.subheader("Model Performance Metrics")
    st.metric(label="Accuracy", value=f"{acc:.2f}")
    st.metric(label="Precision", value=f"{precision:.2f}")
    st.metric(label="Recall", value=f"{recall:.2f}")
    st.metric(label="F1 Score", value=f"{f1:.2f}")

    st.markdown("---")
    st.subheader("Classification Report")
    st.text(classification_report(Y_test, X_test_Prediction))

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(Y_test, X_test_Prediction)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df.style.background_gradient(cmap='Blues'))

# ===============================================================
# 🧍 Passenger Survival Prediction
# ===============================================================
elif menu == "🧍 Passenger Survival Prediction":
    st.title("🧍 Passenger Survival Prediction")

    st.markdown("Provide passenger details below to predict survival:")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 29)
        sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=8, value=0)
    with col2:
        parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=6, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
        embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])

    # Encode inputs
    sex = 0 if sex == "male" else 1
    embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

    # Prepare input
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    if st.button("🔍 Predict Survival"):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.error("❌ The passenger would NOT survive.")
        else:
            st.success("✅ The passenger WOULD survive.")

    st.markdown("---")
    st.caption("💡 Tip: Try different combinations to see how they affect survival probability!")
