import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import pyrebase
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Firebase configuration


# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# User authentication
def login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state['authenticated'] = True
        st.session_state['user'] = user
        st.sidebar.success("Logged in as " + email)
    except Exception as e:
        error_message = e.args[1]
        if 'INVALID_EMAIL' in error_message:
            st.sidebar.error("Invalid email format or the email does not exist.")
        elif 'EMAIL_NOT_FOUND' in error_message:
            st.sidebar.error("Email not found. Please sign up first.")
        elif 'INVALID_PASSWORD' in error_message:
            st.sidebar.error("Invalid password. Please try again.")
        else:
            st.sidebar.error(f"Login failed: {error_message}")

def logout():
    st.session_state['authenticated'] = False
    st.sidebar.success("Logged out")

def signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.sidebar.success("User registered successfully")
    except Exception as e:
        error_message = e.args[1]
        if 'EMAIL_EXISTS' in error_message:
            st.sidebar.error("Email already exists. Please log in.")
        elif 'INVALID_EMAIL' in error_message:
            st.sidebar.error("Invalid email format.")
        elif 'WEAK_PASSWORD' in error_message:
            st.sidebar.error("Weak password. Password should be at least 6 characters.")
        else:
            st.sidebar.error(f"Signup failed: {error_message}")

# Sidebar for login/logout/signup
if st.session_state['authenticated']:
    st.sidebar.write("Logged in as " + st.session_state['user']['email'])
    if st.sidebar.button("Logout"):
        logout()
else:
    st.sidebar.header("Login")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        login(email, password)
    st.sidebar.header("Signup")
    new_email = st.sidebar.text_input("New Email")
    new_password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Signup"):
        signup(new_email, new_password)

# Load pre-trained model
model_path = 'C:/Users/SSNiTHAR/Desktop/assignment/fruit_model.h5'  # Ensure the model file is in the correct directory
model = tf.keras.models.load_model(model_path)

# Function to preprocess and predict fruit name
def predict_fruit(img):
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    fruit_name = decoded_predictions[0][1]
    return fruit_name

# Main app content
if st.session_state['authenticated']:
    st.title("Fruit Identification App")

    option = st.sidebar.selectbox("Select a Page", ["Home", "Upload Image", "Train Model", "Visualize Data"])

    if option == "Home":
        st.write("Welcome to the Fruit Identification App!")
        st.image("C:/Users/SSNiTHAR/Desktop/assignment/fruit.png", caption="Fruit Identification", use_column_width=True)

    elif option == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            prediction = predict_fruit(img)
            st.write(f"Prediction: {prediction}")

    elif option == "Train Model":
        st.header("Train the Model")
        st.write("Upload a dataset to train the model")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())

            # Preprocessing and splitting the data
            X = data.drop("label", axis=1)
            y = data["label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training the model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"Model Accuracy: {accuracy}")

    elif option == "Visualize Data":
        st.header("Data Visualization")
        st.write("Upload a dataset to visualize")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())

            # Display a pairplot
            st.write("Pairplot of the dataset")
            pairplot_fig = sns.pairplot(data, hue="label")
            st.pyplot(pairplot_fig)

            # Display a correlation heatmap
            st.write("Correlation heatmap")
            corr = data.corr()
            heatmap_fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, ax=ax)
            st.pyplot(heatmap_fig)

    # Progress bar demonstration
    st.sidebar.header("Progress")
    progress_bar = st.sidebar.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        # Simulate some process
        import time
        time.sleep(0.01)

    st.sidebar.success("Task Completed!")
else:
    st.title("Please login to access the Fruit Identification App")

