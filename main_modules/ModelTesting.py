## Testing the model
## select the model
import torch
import random
import AgeGenderNet
import FairFaceMulti
import torch, random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL   import Image
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


## we have to define the labels and encoding
age_encoder = LabelEncoder()
age_encoder.fit(['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+'])
gender_encoder = LabelEncoder()
gender_encoder.fit(['Female', 'Male']) # Adjust if your classes are different
race_encoder = LabelEncoder()
race_encoder.fit(['Black', 'East Asian', 'Indian', 'Latino', 'Middle Eastern', 'Southeast Asian', 'White']) # Adjust order/names if needed

## class lists
AGE_CLASSES_LIST = age_encoder.classes_.tolist()
GENDER_CLASSES_LIST = gender_encoder.classes_.tolist()
RACE_CLASSES_LIST = race_encoder.classes_.tolist()


model = AgeGenderNet.AgeGenderNet()
model.load_state_dict(torch.load(r"FairFaceMain\model_checkpoints\model_epoch_10.pth"))
model.eval()                      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@torch.inference_mode()                  # no gradients, saves memory
def predict_img(img_path: str, age, gender, race):
    img = Image.open(img_path).convert("RGB")
    x   = FairFaceMulti.val_tfms(img).unsqueeze(0).to(device)     # shape (1,3,224,224)

    out = model(x)                       # tuple OR dict, depending on your net
    age_id    = out["age"].argmax(1).item()
    gender_id = out["gender"].argmax(1).item()
    race_id   = out["race"].argmax(1).item()
    
    # print("ID INFO")
    # print(f"Age ID: {age_id}, Gender ID: {gender_id}, Race ID: {race_id}")

    # class-name look-up tables (same order you used to encode)
    age_classes = list(age)
    gender_classes = list(gender)
    race_classes = list(race)
    
    # print(age_classes)
    
    pred_age = age_classes[age_id] if age_id < len(age_classes) else "Unknown"
    pred_gender = gender_classes[gender_id] if gender_id < len(gender_classes) else "Unknown"
    pred_race = race_classes[race_id] if race_id < len(race_classes) else "Unknown"

    return pred_age, pred_gender, pred_race

## test with batches
def model_test_batch(valFilePath, test_path, age, gender, race):
    # valFilePath = "FairFace/fairface_label_test.csv"
    valSet = pd.read_csv(valFilePath)
    # test_path = r"FairFace\test"

    # get all files in the test directory, in format: FairFace/test/xxxx.jpg
    test_files = {f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

    
    # extract values of age_range, gender, and race from the valSet and save it to a list for accuracy comparison later
    # this should only already contain the files that are in the test directory
    # age list should be: [3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+, ...] for each image
    
    true_age_list = []
    true_gender_list = []
    true_race_list = []
    pred_age_list = []
    pred_gender_list = []
    pred_race_list = []
    
    # # loop through the test files and get the predictions
    # for file in test_files:
    #     pred_age, pred_gender, pred_race = predict_img(file, age, gender, race)
    #     # print(pred_age, pred_gender, pred_race)
    #     pred_age_list.append(pred_age), pred_gender_list.append(pred_gender), pred_race_list.append(pred_race)
    
    for idx, row in valSet.iterrows():
        file_basename = os.path.basename(row['file']) # Get just the filename
        file_full_path = os.path.join(test_path, file_basename)

        if file_basename in test_files:
            # Predict image, passing the original mapping lists
            pred_age, pred_gender, pred_race = predict_img(file_full_path, age, gender, race)

            if pred_age is not None: # Check if prediction was successful
                # Append true labels
                true_age_list.append(row['age'])
                true_gender_list.append(row['gender'])
                true_race_list.append(row['race'])
                # Append predictions
                pred_age_list.append(pred_age)
                pred_gender_list.append(pred_gender)
                pred_race_list.append(pred_race)
            # else: prediction failed, message already printed in predict_img
        else:
            print(f"Warning: File '{file_basename}' listed in '{valFilePath}' not found in directory '{test_path}'. Skipping.")
    
    # Calculate and print accuracies
    pred_age_list = [pred[0] for pred in pred_age_list]
    pred_gender_list = [pred[0] for pred in pred_gender_list]
    pred_race_list = [pred[0] for pred in pred_race_list]        
        
    age_acc = accuracy_score(true_age_list, pred_age_list)
    gender_acc = accuracy_score(true_gender_list, pred_gender_list)
    race_acc = accuracy_score(true_race_list, pred_race_list)
    print("="*50)
    print(f"\nAccuracy Scores ({len(true_age_list)} images tested):")
    print(f"  Age:    {age_acc:.2%}")
    print(f"  Gender: {gender_acc:.2%}")
    print(f"  Race:   {race_acc:.2%}")
    print("="*50)


    # print(pred_age_list)
    # print accuracies
    # print(f"Age accuracy: {accuracy_score(true_age_list, pred_age_list)}\n"
    #       f"Gender Accuracy: {accuracy_score(true_gender_list, pred_gender_list)}\n"
    #       f"Race Accuracy: {accuracy_score(true_race_list, pred_race_list)}")
    
def testDiffTestSizes(testSizes = [100, 200, 300, 400, 500]):
       # Loop through each test size
    for size in testSizes:
        print(f"Testing with {size} images:")
        testSet.createTestSet(size)
        model_test_batch()
        print("\n" + "="*50 + "\n")  # Separator for readability
    