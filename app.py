import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection App")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Check for 'Fraud' column
    if 'Fraud' not in df.columns:
        st.error("Your CSV must contain a column named 'Fraud' as target!")
    else:
        # Step 2: Automatic preprocessing (no UI)
        df_proc = df.copy()
        le = LabelEncoder()
        for col in df_proc.select_dtypes(include='object').columns:
            df_proc[col] = le.fit_transform(df_proc[col])

        # Step 3: Train model
        X = df_proc.drop('Fraud', axis=1)
        y = df_proc['Fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")

        # Step 4: Predict
        df['Prediction'] = model.predict(X)
        df['Prediction'] = df['Prediction'].map({1: 'Fraud', 0: 'Not Fraud'})

        st.subheader("Predictions")
        st.dataframe(df)

        # Download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

        # ------------------ DATA VISUALIZATION ------------------
        st.subheader("Data Visualizations")

        # 1. Fraud vs Non-Fraud count
        st.write("### Fraud vs Non-Fraud Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Fraud', palette='Set2', ax=ax)
        ax.set_xticklabels(['Not Fraud', 'Fraud'])
        st.pyplot(fig)

        # 2. TotalAmount distribution
        st.write("### TotalAmount Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['TotalAmount'], bins=15, kde=True, color='skyblue', ax=ax2)
        st.pyplot(fig2)

        # 3. Fraud by Product
        st.write("### Fraud Count by Product")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df, x='Product', hue='Fraud', palette='Set1', ax=ax3)
        st.pyplot(fig3)

        # 4. Fraud over time
        if 'Date' in df.columns:
            st.write("### Fraud Over Time")
            df['Date'] = pd.to_datetime(df['Date'])
            df_time = df.groupby(['Date', 'Fraud']).size().reset_index(name='Count')
            fig4, ax4 = plt.subplots(figsize=(10,4))
            sns.lineplot(data=df_time, x='Date', y='Count', hue='Fraud', marker='o', ax=ax4)
            st.pyplot(fig4)
