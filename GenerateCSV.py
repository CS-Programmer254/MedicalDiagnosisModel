import csv
import random

# Dictionary of symptoms for each disease
symptoms_dict = {
    "Typhoid": ["Fever", "Headache"],
    "Malaria": ["Fever", "Vomiting"],
    "HIV": ["Feeble", "Emaciated"],
    "Jaundice": ["Jaundice", "Severe vomiting"]
}

# Function to generate random symptoms and disease
def generate_random_input():
    disease = random.choice(list(symptoms_dict.keys()))
    symptoms = symptoms_dict[disease]
    return symptoms, disease

# Main function to generate the CSV file
def generate_csv(file_path, num_samples):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symptoms', 'Disease'])  # Write header row
        for _ in range(num_samples):
            symptoms, disease = generate_random_input()
            writer.writerow([', '.join(symptoms), disease])

# Generate CSV file with 1000 samples
file_path = 'diagnosis_data.csv'
num_samples = 1000
generate_csv(file_path, num_samples)
print(f"CSV file '{file_path}' with {num_samples} samples created successfully.")
