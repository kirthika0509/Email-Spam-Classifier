import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Configure the web page
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Page title
st.title("📧 Email Spam Classifier")

st.write(
    "Enter a message below to check whether it is Spam or Ham."
)

st.info(
    "Note: This prototype is trained on the SMS Spam Collection dataset. "
    "Predictions on actual emails may be less reliable."
)

# Input fields
subject = st.text_input("Email Subject (Optional)")

message = st.text_area(
    "Message Content",
    height=180,
    placeholder="Enter your message here..."
)

# Prediction button
if st.button("🔍 Check Message", use_container_width=True):

    combined_message = (subject + " " + message).strip()

    if not combined_message:
        st.warning("Please enter a subject or message.")

    else:
        # Convert text into numerical features
        message_tfidf = vectorizer.transform([combined_message])

        # Predict the message category
        prediction = model.predict(message_tfidf)[0]

        # Display the result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE DETECTED!")
            st.write("This message has been classified as Spam.")

        else:
            st.success("✅ HAM — NOT IDENTIFIED AS SPAM")
            st.write("This message has been classified as Ham.")

# Example messages
st.divider()

st.subheader("🧪 Try Example Messages")

if st.button("Try Spam Example"):
    example = "Congratulations! You have won a free prize. Claim now!"
    st.write("**Example:**", example)

    example_tfidf = vectorizer.transform([example])
    result = model.predict(example_tfidf)[0]

    if result == "spam":
        st.error("🚨 SPAM MESSAGE DETECTED!")
    else:
        st.success("✅ HAM — NOT IDENTIFIED AS SPAM")

if st.button("Try Ham Example"):
    example = "Hi, how are you? Please send me the class notes."
    st.write("**Example:**", example)

    example_tfidf = vectorizer.transform([example])
    result = model.predict(example_tfidf)[0]

    if result == "spam":
        st.error("🚨 SPAM MESSAGE DETECTED!")
    else:
        st.success("✅ HAM — NOT IDENTIFIED AS SPAM")

st.divider()

st.caption("Machine Learning Project | Python | Scikit-learn | Streamlit")