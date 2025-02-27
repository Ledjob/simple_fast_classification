import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# # get data files
# !wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
# !wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)
    return df

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_data = load_data(train_file_path)
test_data = load_data(test_file_path)

# Feature extraction - convert the text messages into numerical features
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data['message'])
y_train = train_data['label']

# Additional features for training data
def extract_features(messages):
    features = pd.DataFrame()
    # Length of message
    features['length'] = messages.apply(len)
    # Count of digits
    features['digit_count'] = messages.apply(lambda x: sum(c.isdigit() for c in x))
    # Count of special characters
    features['special_char_count'] = messages.apply(lambda x: sum(c in "!@#$%^&*()-+=<>?/|\\" for c in x))
    # Count of uppercase words (often used in spam)
    features['uppercase_count'] = messages.apply(lambda x: sum(w.isupper() for w in x.split() if len(w) > 1))
    # Count of spam indicators
    spam_indicators = ["free", "win", "won", "prize", "cash", "call", "text", "urgent", "claim", "click", "offer", "limited"]
    features['spam_indicator_count'] = messages.apply(lambda x: sum(w.lower() in spam_indicators for w in x.split()))
    # Presence of phone numbers (crude check)
    features['has_phone'] = messages.apply(lambda x: 1 if sum(c.isdigit() for c in x) > 8 else 0)
    # Money symbols
    features['has_money_symbol'] = messages.apply(lambda x: 1 if any(c in x for c in "£$€") else 0)
    
    return features

# Extract additional features
train_additional_features = extract_features(train_data['message'])

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Function to predict if a message is spam or ham
def predict_message(message):
    # Transform the message using the vectorizer
    message_counts = vectorizer.transform([message])
    
    # Extract additional features
    message_features = extract_features(pd.Series([message]))
    
    # Get the probability of spam
    spam_probability = clf.predict_proba(message_counts)[0][1]
    
    # Adjust probability based on additional features
    if message_features['has_phone'].values[0] == 1:
        spam_probability += 0.2
    if message_features['has_money_symbol'].values[0] == 1:
        spam_probability += 0.2
    if message_features['spam_indicator_count'].values[0] > 0:
        spam_probability += 0.15 * message_features['spam_indicator_count'].values[0]
    if message_features['uppercase_count'].values[0] > 1:
        spam_probability += 0.1
    
    # Ensure probability is between 0 and 1
    spam_probability = min(max(spam_probability, 0), 1)
    
    # Return the result
    label = 'spam' if spam_probability > 0.5 else 'ham'
    return [spam_probability, label]

# Test function (do not modify)
def test_predictions():
    test_messages = ["how are you doing today",
                    "sale today! to stop texts call 98912460324",
                    "i dont want to go. can we try it a different day? available sat",
                    "our new mobile video service is live. just install on your phone to start watching.",
                    "you have won £1000 cash! call to claim your prize.",
                    "i'll bring it tomorrow. don't forget the milk.",
                    "wow, is your arm alright. that happened to me one time too"
                   ]
    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    
    passed = True
    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        print(f"Message: {msg}")
        print(f"Prediction: {prediction[1]}, Expected: {ans}")
        print(f"Probability: {prediction[0]:.4f}")
        print("---")
        if prediction[1] != ans:
            passed = False
    
    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

# Run the test
test_predictions()