import pandas as pd
from keras import Sequential
from keras.src.saving import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
import pickle  # For loading the trained vectorizer

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('diagnosis_data.csv')

# Preprocessing
X = data['Symptoms']  # Symptoms as input features
y = data['Disease']  # Disease as target variable
# Convert symptoms to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Encode the diagnosis labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Model building
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_vectorized.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
    # Output layer with softmax activation for multi-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('diagnosis_model.h5')

# Save the trained vectorizer (assuming pickle is used)
with open('diagnosis_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Load the trained model (unchanged)
model = load_model('diagnosis_model.h5')

# Load the trained vectorizer
with open('diagnosis_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def preprocess_input(symptoms, vectorizer):
    symptoms_str = ', '.join(symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms_str]).toarray()
    return symptoms_vectorized


def make_diagnosis(symptoms, model, label_encoder, vectorizer):
    symptoms_vectorized = preprocess_input(symptoms, vectorizer)
    prediction = model.predict(symptoms_vectorized)
    predicted_disease = label_encoder.inverse_transform([prediction.argmax()])[0]
    return predicted_disease


# Example usage
user_input = input("Enter symptoms (separated by commas): ")
symptoms = user_input.strip().split(',')
diagnosis = make_diagnosis(symptoms, model, label_encoder, vectorizer)
print(f"The predicted disease is: {diagnosis}")
