import sys
import os

# Add the model_code folder to the system path
sys.path.append(os.path.join(os.getcwd(), "model_code"))

# Import functions from the correct scripts
from preprocessing import preprocess_data  # Replace with actual function name in preprocessing.py
from train_model import train_model  # Replace with actual function name in train_model.py
from test_model import test_model  # Replace with actual function name in test_model.py

def main():
    # Step 1: Preprocess the data
    print("Preprocessing the data...")
    preprocess_data()  # Call the preprocessing function
    
    # Step 2: Train the model
    print("Training the model...")
    train_model()  # Call the training function
    
    # Step 3: Test the model
    print("Testing the model...")
    test_model()  # Call the testing function

if __name__ == "__main__":
    main()
