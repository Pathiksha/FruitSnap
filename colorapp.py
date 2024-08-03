import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import time
import firebase_admin
from firebase_admin import credentials, auth
import streamlit.components.v1 as components

# Initialize Firebase Admin SDK
firebaseConfig = {
  "apiKey": "AIzaSyC3tOnIbRKbEUKryqWFjDA9aIioXgjHEL0",
  "authDomain": "colorapp-280ef.firebaseapp.com",
  "databaseURL": "https://colorapp-280ef-default-rtdb.firebaseio.com",
  "projectId": "colorapp-280ef",
  "storageBucket": "colorapp-280ef.appspot.com",
  "messagingSenderId": "442382208499",
  "appId": "1:442382208499:web:d631911dc72283db3e9686",
  "measurementId": "G-4CJ44EG8L6"
};

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()



# Database helper functions
def create_connection():
    return sqlite3.connect('users.db')

def create_db():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Initialize the database (only needs to be run once)
create_db()

# Streamlit app
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        st.sidebar.write(f"Logged in as {st.session_state.username}")

        # Sidebar for color visualization options
        st.sidebar.title("Data Visualization Options")
        visualization_option = st.sidebar.selectbox("Choose Visualization Type", ["None", "Color Blocks", "Bar Chart"])

        # Main page content
        st.title("Color Identification App")
        st.text("Upload an image and identify the prominent colors.")

        # Image upload on the main page
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        num_clusters = st.slider("Number of Colors", 1, 20, 5)

        colors = None  # Initialize colors variable

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            with st.spinner('Processing...'):
                time.sleep(2)
                st.success('Processing done!')

            def get_colors(image, num_colors):
                img_array = np.array(image)
                img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
                kmeans = KMeans(n_clusters=num_colors)
                kmeans.fit(img_array)
                colors = kmeans.cluster_centers_
                return colors

            colors = get_colors(image, num_clusters)
            st.write(f"Identified {num_clusters} colors:")
            for color in colors:
                st.write(f"Color: {color}")
                st.markdown(f'<div style="width: 50px; height: 50px; background-color: rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"></div>', unsafe_allow_html=True)

            if visualization_option == "Color Blocks":
                # Generate color block visualization
                fig, ax = plt.subplots()
                colors = np.array(colors).astype(int)
                for idx, color in enumerate(colors):
                    ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=np.array(color)/255))
                plt.xlim(0, num_clusters)
                plt.ylim(0, 1)
                plt.axis('off')
                st.sidebar.pyplot(fig)

            elif visualization_option == "Bar Chart":
                # Generate bar chart visualization
                if colors is not None:
                    fig, ax = plt.subplots()
                    color_names = [f"Color {i+1}" for i in range(num_clusters)]
                    color_values = np.linalg.norm(colors, axis=1)  # Magnitude of colors for bar chart
                    # Convert RGB colors to Matplotlib color format
                    color_mapped = [tuple(c / 255) for c in colors]
                    ax.bar(color_names, color_values, color=color_mapped)
                    plt.xlabel('Colors')
                    plt.ylabel('Magnitude')
                    plt.title('Color Magnitude Bar Chart')
                    st.sidebar.pyplot(fig)

        # Logout functionality
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.pop('authenticated', None)  # Optional: Ensure session state is cleared
            st.experimental_rerun()  # Alternative to manually refreshing if st.experimental_rerun() is available

    else:
        # Display Firebase authentication component
        st.sidebar.title("Authentication")
        html_code = """
        <iframe src="firebase_auth.html" width="100%" height="400px"></iframe>
        """
        components.html(html_code)

        # Handle authentication state change from JavaScript
        st.session_state['auth_status'] = st.empty()
        st.session_state['auth_status'].write("Waiting for authentication...")

        # Listen for messages from JavaScript
        def on_message(message):
            if message['type'] == 'LOGIN_SUCCESS':
                st.session_state.authenticated = True
                st.session_state.username = message['user']['email']
                st.experimental_rerun()
            elif message['type'] == 'LOGIN_FAILED':
                st.session_state.auth_status.write("Login failed: " + message['error'])
            elif message['type'] == 'SIGNUP_SUCCESS':
                st.session_state.auth_status.write("Signup successful! Please login.")
            elif message['type'] == 'SIGNUP_FAILED':
                st.session_state.auth_status.write("Signup failed: " + message['error'])
            elif message['type'] == 'LOGOUT_SUCCESS':
                st.session_state.authenticated = False
                st.session_state.username = None
                st.experimental_rerun()

        # Register the on_message function to handle incoming messages
        components.html(html_code, height=600, on_message=on_message)

if __name__ == '__main__':
    main()
