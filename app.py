from flask import Flask, render_template, request
import numpy as np
import pickle
import sys
import os

# --- MOCKUP DATA FOR DEMONSTRATION ---
# In a real scenario, this data would come from your 'class_svc.pkl' file.
# The feature names MUST be in the EXACT order your model expects.
# You MUST replace the 'Dummy' placeholder names with your actual 25 one-hot encoded features.
expected_feature_names = [
    'Gender', 'Age', 'Number_of_Failures', 'Final_Grade', 'Absences', 
    'Study_Time', 'Parental_Status_A', 'Parental_Status_B', 'Activities',
    # 25 other One-Hot Encoded features from other columns (e.g., School_GP, Family_Size, etc.)
    'Dummy_School_A', 'Dummy_School_B', 'Dummy_Family_Size_1', 'Dummy_Family_Size_2', 
    'Dummy_Mjob_1', 'Dummy_Mjob_2', 'Dummy_Mjob_3', 'Dummy_Mjob_4', 'Dummy_Mjob_5',
    'Dummy_Fjob_1', 'Dummy_Fjob_2', 'Dummy_Fjob_3', 'Dummy_Fjob_4', 'Dummy_Fjob_5',
    'Dummy_Reason_1', 'Dummy_Reason_2', 'Dummy_Reason_3', 'Dummy_Reason_4',
    'Dummy_Guardian_1', 'Dummy_Guardian_2', 'Dummy_TravelTime_1', 'Dummy_TravelTime_2',
    'Dummy_Health_1', 'Dummy_Health_2', 'Dummy_Health_3'
] # Total of 34 features (9 explicit + 25 dummy)

# --- NEW: Get the exact feature count dynamically ---
FEATURE_COUNT = len(expected_feature_names) 

# A dictionary to map the form input names to the index in the feature list
# This is crucial for correctly populating the input array.
FORM_TO_MODEL_MAP = {
    'Gender': expected_feature_names.index('Gender'),
    'Age': expected_feature_names.index('Age'),
    'Number_of_Failures': expected_feature_names.index('Number_of_Failures'),
    'Final_Grade': expected_feature_names.index('Final_Grade'),
    'Absences': expected_feature_names.index('Absences'),
    'Study_Time': expected_feature_names.index('Study_Time'),
    'Activities': expected_feature_names.index('Activities'),
    # Parental_Status is handled specially in predict()
}

# --- END MOCKUP DATA ---

# Load the trained model and associated data (e.g., feature names)
try:
    with open('class_svc.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Extract model (assuming the model is saved directly or inside a dictionary key)
    if isinstance(data, dict):
        model = data.get("model") or data.get("svc") or data.get("pipeline") or data[list(data.keys())[0]]
        # If your actual 34 feature names are stored in the pickle, load them here
        # expected_feature_names = data.get("feature_names", expected_feature_names) 
    else:
        model = data
    
    # --- REMOVED WARNING CHECK ---
    # The misleading warning check has been removed. The logic now uses 
    # the dynamically determined FEATURE_COUNT (34) for prediction.
        
except FileNotFoundError:
    model = None
    print("Error: 'class_svc.pkl' not found. Please ensure the model file is in the same directory.", file=sys.stderr)
except Exception as e:
    model = None
    print(f"Error loading model from pickle: {e}", file=sys.stderr)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model could not be loaded. Check server logs.")

    try:
        # 1. Collect all input values from form
        form_inputs = {
            'Gender': float(request.form['Gender']), # Assuming 0/1 encoding
            'Age': float(request.form['Age']),
            'Number_of_Failures': float(request.form['Number_of_Failures']),
            'Final_Grade': float(request.form['Final_Grade']),
            'Parental_Status': float(request.form['Parental_Status']), # Needs special handling below
            'Absences': float(request.form['Absences']),
            'Study_Time': float(request.form['Study_Time']),
            'Activities': float(request.form['Activities']),
        }
        
        # --- IMPORTANT FIX HERE: Create the dynamically sized feature array ---
        
        # Initialize an array of FEATURE_COUNT zeros. This makes the array size 
        # match the length of the expected_feature_names list (which is currently 34).
        final_features = np.zeros(FEATURE_COUNT) 

        # Map the input values to the correct index in the feature array.
        for name, value in form_inputs.items():
            if name in FORM_TO_MODEL_MAP:
                # For simple numerical features, place the value directly
                index = FORM_TO_MODEL_MAP[name]
                final_features[index] = value
            
            # Special handling for categorical features (like Parental_Status) 
            # that were one-hot encoded into multiple columns.
            elif name == 'Parental_Status':
                # Example: If Parental_Status=1 maps to 'Parental_Status_A' feature
                if value == 1:
                    index = expected_feature_names.index('Parental_Status_A')
                    final_features[index] = 1.0 # Set the correct dummy variable to 1
                # Example: If Parental_Status=2 maps to 'Parental_Status_B' feature
                elif value == 2:
                    index = expected_feature_names.index('Parental_Status_B')
                    final_features[index] = 1.0 # Set the correct dummy variable to 1
                # Add more conditions for other categories if needed

        # The final input must be a 2D array: (1, FEATURE_COUNT)
        final_features = final_features.reshape(1, -1)

        # Predict using model
        pred = model.predict(final_features)
        
        # The SVC model returns the class label (0 or 1)
        output = "YES (Student may Dropout)" if pred[0] == 1 else "NO (Student will Continue)"

        return render_template('index.html', prediction_text=f"Prediction: {output}")

    except ValueError:
        return render_template('index.html', prediction_text="Error: Please ensure all fields are filled with valid numbers.")
    except Exception as e:
        # Print the error to the console for debugging
        print(f"Prediction Error: {e}", file=sys.stderr)
        return render_template('index.html', prediction_text=f"Prediction Error. Please check server logs for details.")

if __name__ == "__main__":
    app.run(debug=True)