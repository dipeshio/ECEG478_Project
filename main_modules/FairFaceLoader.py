## loading in all of the important libraries
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import tqdm as tqdm
import shutil
import os, time, datetime
from statistics import mean


import AgeGenderNet as AgeGenderNet
import FairFaceMulti as FairFaceMulti
import ModelTesting as ModelTesting


## We're going to create a data loader class because this is NEEDED and I am losing it

class FairFaceLoader():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.balanced_df = None
        self.secondary_df = None
        self.sampleSet = None
        self.secondary_sampleSet = None
        self.testSet = None
        
        ## 
        self.trainSet = None
        self.valSet = None
        self.testSet = None
        
    def __print__(self):
    # function that prints the first 5 rows of the dataframe
        print(self.df.head())
        
    def __len__(self):
    # function that returns the length of the dataframe
        return len(self.df)
    
    def clean(self):
        # print(self.df.head())
        if 'service_test' in self.df.columns:
            self.df = self.df.drop(columns='service_test')
        else:
            print("Column 'service_test' not in df; could have been removed already.")    
        ## Combining the age bins of '60-69' and 'more than 70' into '60+' and relabeling the age bins
        self.df['age'] = self.df['age'].replace({'60-69': '60+', 'more than 70': '60+'})
        # rename Latino_Hispanic to Latino
        self.df['race'] = self.df['race'].replace({'Latino_Hispanic': 'Latino'})
        self.df = self.df.drop_duplicates(subset=['file'], keep='first')
        
        # encoding categorical labels
        age_bins = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']
        self.df = self.df[self.df['age'].isin(age_bins)]  # ensure only those bins are used
        
        ## these will be used to check what the encoding and decoding looks like

        self.age_encoder = LabelEncoder()
        self.df['age_label'] = self.age_encoder.fit_transform(self.df['age'])
        self.gender_encoder = LabelEncoder()
        self.df['gender_label'] = self.gender_encoder.fit_transform(self.df['gender'])
        self.race_encoder = LabelEncoder()
        self.df['race_label'] = self.race_encoder.fit_transform(self.df['race'])
        
        ## drop the original columns
        # self.df = self.df.drop(['age', 'gender','race'], axis=1)
        # print(self.df.head())
        return self.df
    
    def cleanForTraining(self, dataframe):
        dataframe['age_label'] = self.age_encoder.fit_transform(dataframe['age'])
        dataframe['gender_label'] = self.gender_encoder.fit_transform(dataframe['gender'])
        dataframe['race_label'] = self.race_encoder.fit_transform(dataframe['race'])
        
        
    
    def balance(self, n_per_group):
        self.clean()
        # Ensure the dataframe has the required columns
        if not {'age_label', 'gender_label', 'race_label'}.issubset(self.df.columns):
            raise ValueError("Columns age_label, gender_label, race_label must exist. Call clean() first.")
        
        attr_cols = ['age_label', 'gender_label', 'race_label']
        state_value = np.random.randint(0, 1500)
        grouped = self.df.groupby(attr_cols)
        # print("Grouped dataframe:")
        # print(grouped.head())
        
        self.balanced_df = grouped.apply(
            lambda x: x.sample(n=min(len(x), n_per_group), random_state=state_value), 
            include_groups=False
        ).reset_index(drop=True)
        
        return self.balanced_df
    
    def randomizeSet(self):
        # function that randomizes the dataset
        self.clean()
        state_value = np.random.randint(0, 1500) # Use numpy directly to generate a random integer
        ## only take 80% of the dataset to randomize
        self.df = (self.df).sample(frac=0.8, random_state=state_value).reset_index(drop=True)
    
    def randomizeBalancedSet(self):
        # function that randomizes the dataset
        state_value = np.random.randint(0, 1500) # Use numpy directly to generate a random integer
        self.tempSet = self.balanced_df.sample(frac=1, random_state=state_value).reset_index(drop=True)
        self.balanced_df = self.tempSet
    
    def displayChart(self, name):
        if name == "original":
            name = self.df
        elif name == "balanced":
            name = self.balanced_df
        else: 
            print("Please select either 'original' or 'balanced' dataset")
            return        
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(name['age_label'].value_counts().index, name['age_label'].value_counts().values)
        plt.title('Age Distribution')
        plt.xlabel('Age Label')
        plt.ylabel('Count')

        ## count how many are label 7 and 8
        print(name['age_label'].value_counts())
        # 3270 as a combination of 7 and 8 before combinations
        # 

        ## Gender labels
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(name['gender_label'].value_counts().index, name['gender_label'].value_counts().values)
        plt.title('Gender Distribution')
        plt.xlabel('Gender Label')
        plt.ylabel('Count')

        ## Race labels
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(name['race_label'].value_counts().index, name['race_label'].value_counts().values)
        plt.title('Race Distribution')
        plt.xlabel('Race Label')
        plt.ylabel('Count')
        
    
    def getDecodedLabels(self):
        
        return (
            list(zip(self.age_encoder.classes_, range(len(self.age_encoder.classes_)))),
            list(zip(self.gender_encoder.classes_, range(len(self.gender_encoder.classes_)))),
            list(zip(self.race_encoder.classes_, range(len(self.race_encoder.classes_))))
        )
        
    
    def random_int(self):
    # function that generates a random int between two input ranges
        return random.randint(0,1500)
    
    def createTrainValSet(self, type, n_per_group=100):
        self.secondary_df = self.balance(200)
        self.remainingSet = None
        # print("Secondary dataframe:")
        # print(self.secondary_df.head())

        
        # if type == "train":
        self.trainSet = self.secondary_df.sample(frac=0.7, random_state=self.random_int()).reset_index(drop=True)
        self.cleanForTraining(self.trainSet)
        
        ## dropping the values in the train set from the secondary dataframe  
        self.remainingSet = self.secondary_df[~self.secondary_df['file'].isin(self.trainSet['file'])]

        # print("Remaining set size:")
        # print(len(self.remainingSet))
        # print("Remaining set:")
        # print(self.remainingSet)
        
        # self.trainSet.reset_index(drop=True, inplace=True)

        # elif type == "val":
        self.valSet = self.remainingSet.sample(frac=0.50, random_state=self.random_int()).reset_index(drop=True)
        self.cleanForTraining(self.valSet)

            
        self.sampleSet = self.secondary_df.sample(frac=1, random_state=self.random_int()).reset_index(drop=True)
        
        source_base_dir = "FairFace" ## where it's coming from
        train_dir = "FairFaceMain/train"   ## where it'll go
        val_dir = "FairFaceMain/val"
        train_csv_out_path = "FairFaceMain/fairface_label_train.csv"
        val_csv_out_path = "FairFaceMain/fairface_label_val.csv"
        
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        
        ## remove all files in the train and val directories
        for filename in os.listdir(train_dir):
            file_path = os.path.join(train_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
                
        for filename in os.listdir(val_dir):
            file_path = os.path.join(val_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
                
                
        train_files = self.trainSet['file'].tolist()
        val_files = self.valSet['file'].tolist()
        
        for rel_path in train_files:
            src = os.path.join(source_base_dir, rel_path)
            dst = os.path.join(train_dir, os.path.basename(rel_path))

            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)  # keep metadata
                except Exception as e:
                    print(f"[WARN] could not copy {src}: {e}")
            else:
                print(f"[WARN] missing source file {src}")
        
        for rel_path in val_files:
            src = os.path.join(source_base_dir, rel_path)
            dst = os.path.join(val_dir, os.path.basename(rel_path))

            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)  # keep metadata
                except Exception as e:
                    print(f"[WARN] could not copy {src}: {e}")
            else:
                print(f"[WARN] missing source file {src}")

        print("Finished copying files.")
        
        filesTrain = os.listdir(train_dir)
        filesVal = os.listdir(val_dir)
        
        
        ## get the file names without the dir
        filesTrain = [os.path.basename(file) for file in filesTrain]
        filesVal = [os.path.basename(file) for file in filesVal]
        
        basenamesTrain = [os.path.basename(p) for p in train_files]
        basenamesVal = [os.path.basename(p) for p in val_files]
        
        traindf_labels = self.trainSet[
            self.trainSet["file"].str.split("/").str[-1].isin(basenamesTrain)].copy()
        
        valdf_labels = self.valSet[
            self.valSet["file"].str.split("/").str[-1].isin(basenamesVal)].copy()
        
        
        # ## replace file dir for valSet --> was: train/xxxx.jpg, now: val/xxxx.jpg
        traindf_labels['file'] = traindf_labels['file'].str.replace('train/', 'train/')
        valdf_labels['file'] = valdf_labels['file'].str.replace('train/', 'val/')
        
        
        # new sort numerically by the digits in the filename
        traindf_labels["__id"] = (
            traindf_labels["file"]
            .str.extract(r"(\d+)")
            .astype(int)
        )
        traindf_labels.sort_values("__id", inplace=True)
        traindf_labels.drop(columns="__id", inplace=True)
        traindf_labels.reset_index(drop=True, inplace=True)
        
        
        valdf_labels["__id"] = (
            valdf_labels["file"]
            .str.extract(r"(\d+)")
            .astype(int)
        )
        valdf_labels.sort_values("__id", inplace=True)
        valdf_labels.drop(columns="__id", inplace=True)
        valdf_labels.reset_index(drop=True, inplace=True)
        
        traindf_labels.to_csv(train_csv_out_path, index=False)
        valdf_labels.to_csv(val_csv_out_path, index=False)
        print(f"[DONE] copied {len(traindf_labels)} files and wrote labels to {train_csv_out_path}")

        return self.trainSet, self.valSet
    
    def checkOverlap(self):
        # function that checks for overlap between the train and val sets
        if self.trainSet is None or self.valSet is None:
            print("Train or Val set is not created yet.")
            return
        
        train_files = set(self.trainSet['file'])
        val_files = set(self.valSet['file'])
        
        overlap = train_files.intersection(val_files)
        
        if len(overlap) > 0:
            print(f"Overlap found: {len(overlap)} files")
            
            ## print all of the overlapping files
            for file in overlap:
                print(file)
        else:
            print("No overlap found")
    
        
    def createTestSet(self, test_size):
        print("Sampler method is not yet implemented.")
        #########################################
        self.secondary_sampleSet = self.df.copy()
        source_base_dir = "FairFace" ## where it's coming from
        test_dir = "FairFaceMain/test"   ## where it'll go
        csv_out_path = "FairFaceMain/fairface_label_test.csv"

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            
        ## check what files are in the directory and remove all
        for filename in os.listdir(test_dir):
            file_path = os.path.join(test_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
                
        # get a list of all the files (relative paths like 'val/xxxx.jpg') in the validation set
        
        testFiles = self.secondary_sampleSet['file'].tolist()
        random_files = [] # initalize empty list first
        
        # if len(testFiles) >= test_size:
        #     random_files = random.sample(testFiles, test_size)                  
        # else:
        #     print(f"Warning: Only {len(testFiles)} files available in validation set. Selecting all.")
            
        if len(testFiles) < test_size:
            print(f"[INFO] only {len(testFiles)} files available; using all of them.")
            random_files = testFiles
        else:
            random_files = random.sample(testFiles, test_size)
                    
            # Copy the files to the test directory
        for rel_path in random_files:
            src = os.path.join(source_base_dir, rel_path)
            dst = os.path.join(test_dir, os.path.basename(rel_path))

            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)  # keep metadata
                except Exception as e:
                    print(f"[WARN] could not copy {src}: {e}")
            else:
                print(f"[WARN] missing source file {src}")

        print("Finished copying files.")
        
        ## Now need to create a new CSV file for all of the information regarding the files in the test directory
        # gen idea is to go call the val csv file and drop all rows that do not have the file name in the test directory

        files = os.listdir(test_dir)
        # Get the file names without the dir path
        files = [os.path.basename(file) for file in files]
        
        basenames = [os.path.basename(p) for p in random_files]
        
        
        # keep only rows whose basename matches something we copied
        df_labels = self.secondary_sampleSet[
            self.secondary_sampleSet["file"].str.split("/").str[-1].isin(basenames)
        ].copy()

        # change any leading file paths to "test/"
        df_labels["file"] = df_labels["file"].str.replace(r"^(train|val)/", "test/", regex=True)


        # new sort numerically by the digits in the filename
        df_labels["__id"] = (
            df_labels["file"]
            .str.extract(r"(\d+)")
            .astype(int)
        )
        df_labels.sort_values("__id", inplace=True)
        df_labels.drop(columns="__id", inplace=True)
        df_labels.reset_index(drop=True, inplace=True)
        
        df_labels.to_csv(csv_out_path, index=False)
        print(f"[DONE] copied {len(df_labels)} files and wrote labels to {csv_out_path}")

        return random_files

    
    def getBalancedSet(self):
        return self.balanced_df
    
    def getBalancedSetLength(self):
        if self.balanced_df is None:
            print("Balanced set is not created yet.")
            return 0
        return len(self.balanced_df)
    
    def getDataSet(self):
        return self.df
    
    def getDataSetLength(self):
        if self.df is None:
            print("Data set is not created yet.")
            return 0
        return len(self.df)
    
    def getTrainSet(self):
        return self.trainSet
    
    def getTrainSetLength(self):
        if self.trainSet is None:
            print("Train set is not created yet.")
            return 0
        return len(self.trainSet)
    
    def getValSet(self):
        return self.valSet
    
    def getValSetLength(self):
        if self.valSet is None:
            print("Val set is not created yet.")
            return 0
        return len(self.valSet)
    
    def getTestSet(self):
        return self.testSet
    
    def getTestSetLength(self):
        if self.testSet is None:
            print("Test set is not created yet.")
            return 0
        return len(self.testSet)
        

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    epoch_loss, correct_age, correct_gender, correct_race, n = 0,0,0,0,0 # Initialize n=0
    print(f"Starting {'training' if train else 'validation'} epoch...")
    batch_num = 0
    for imgs, targets in tqdm.tqdm(loader):
        # print(f"Processing batch {batch_num}")
        batch_num += 1
        try:
            imgs = imgs.cuda()
            t_age    = targets["age"].cuda()
            t_gender = targets["gender"].cuda()
            t_race   = targets["race"].cuda() # Get race target

            with torch.set_grad_enabled(train):
                preds = model(imgs)
                loss, parts = AgeGenderNet.criterion(preds, {"age":t_age, "gender":t_gender, "race": t_race}) # Include race in criterion if needed

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, pa = preds["age"].max(1)
            _, pg = preds["gender"].max(1)
            _, pr = preds["race"].max(1) # Use pr for race prediction index

            correct_age    += (pa==t_age).sum().item()
            correct_gender += (pg==t_gender).sum().item()
            correct_race   += (pr==t_race).sum().item() # Calculate correct race predictions

            n += imgs.size(0)
            epoch_loss += loss.item()*imgs.size(0)
        except Exception as e:
            print(f"Error in batch {batch_num-1}: {e}")
            # raise # Re-raise the exception to stop execution
            # continue # Skip this batch and continue

    print(f"Finished {'training' if train else 'validation'} epoch.")
    # Handle potential division by zero if n remains 0
    if n == 0:
        print("Warning: No items processed in the loader.")
        return (0, 0, 0, 0) # Return 4 zeros
    # Return loss and all three accuracies
    return (epoch_loss/n,
            correct_age/n,
            correct_gender/n,
            correct_race/n) # Add race accuracy to the return tuple
    
def testDiffSizes(age, gender, race, testSizes = [100, 200, 300, 400, 500]):
    # Loop through each test size
    for size in testSizes:
        print(f"Testing with {size} images:")
        testSet.createTestSet(size)
        ModelTesting.model_test_batch("FairFaceMain/fairface_label_test.csv", "FairFaceMain/test", age, gender,race)





testSet = FairFaceLoader('FairFace/fairface_label_train.csv')
testSet.clean()
# testSet.randomizeSet()
# # testSet.createTrainValSet('train')
# testSet.createTrainValSet('val')

# # print(testSet.getTrainSet())
# print(testSet.getValSet())

# # testSet.checkOverlap()

# # ## check the lengths of the sets
# # print("Train set length: ", testSet.getTrainSetLength())
# # print("Val set length: ", testSet.getValSetLength())



trainSet = FairFaceLoader('FairFace/fairface_label_val.csv')
trainSet.randomizeSet()
testSet.createTestSet(500)

import os
import time
import datetime
import torch
import pandas as pd
from torch.utils.data import DataLoader
from statistics import mean
import multiprocessing 

if __name__ == "__main__":
    multiprocessing.freeze_support() # need this for some reason
    trainingSet = FairFaceLoader('FairFaceMain/fairface_label_train.csv')
    trainingSetData = trainingSet.getDataSet()

    validationSet = FairFaceLoader('FairFaceMain/fairface_label_val.csv')
    valSetData = validationSet.getDataSet()

    print(valSetData.head())


    train_ds = FairFaceMulti.FairFaceMulti(trainingSetData, "FairFaceMain/train", train=True)
    val_ds   = FairFaceMulti.FairFaceMulti(valSetData,      "FairFaceMain/val",   train=False)


    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # define the model and loss functions
    model = AgeGenderNet.AgeGenderNet().cuda()
    loss_age = nn.CrossEntropyLoss()
    loss_gender = nn.CrossEntropyLoss()     # fairly balanced already
    loss_race = nn.CrossEntropyLoss()     # fairly balanced already

    # AgeGenderNet.criterion()

    ## optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ### use gpu and stuff
    nvidia_smi = os.popen("nvidia-smi").read()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
        
        
    ## location to store models
    if not os.path.exists('FairFaceMain/model_checkpoints'):
        os.makedirs('FairFaceMain/model_checkpoints')

    EPOCHS = 15
    # to keep track of time
    epoch_times = []

    trainAcc = []
    valAcc = []
    trainLoss = []
    valLoss = []
    trainF1 = []
    valF1 = []

    ## start the training loop
    for epoch in range(EPOCHS):
        t0 = time.time()
        
        tr_loss, tr_acc_age, tr_acc_gen, tr_acc_race = run_epoch(train_loader, train=True)
        vl_loss, vl_acc_age, vl_acc_gen, vl_acc_race = run_epoch(val_loader,   train=False)
        scheduler.step()
        # Initialize variables for ETA calculation
        start_time = time.time()
        
        ## adding metrids to the lists
        trainLoss.append(tr_loss)
        valLoss.append(vl_loss)
        
        avg_tr_acc = (tr_acc_age + tr_acc_gen + tr_acc_race) / 3
        avg_vl_acc = (vl_acc_age + vl_acc_gen + vl_acc_race) / 3
        trainAcc.append(avg_tr_acc)
        valAcc.append(avg_vl_acc)

        # Calculate elapsed time and ETA
        epoch_times.append(time.time() - t0)
        avg_epoch   = mean(epoch_times)
        remaining   = EPOCHS - (epoch + 1)
        eta_seconds = avg_epoch * remaining
        eta_str     = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Format ETA as HH:MM:SS
        eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        ## incramentally save the model after every 20 epochs
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"FairFaceMain/model_checkpoints/2model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

        # Print metrics
        print(
            f"{epoch:02d}  "
            f"tr_loss={tr_loss:.3f}  vl_loss={vl_loss:.3f} |\n"
            f"TRAIN  age={tr_acc_age:.2%}  gen={tr_acc_gen:.2%}  race={tr_acc_race:.2%} |\n"
            f"VAL    age={vl_acc_age:.2%}  gen={vl_acc_gen:.2%}  race={vl_acc_race:.2%} |\n"
            f"ETA ~ {eta_str}")
        
    ## print plots of the training and validation loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch+2), trainLoss, label='Train Loss')
    plt.plot(range(1, epoch+2), valLoss, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.show()
## 

# age, gender, race = testSet.getDecodedLabels()
# # result = ModelTesting.predict_img("FairFaceMain/test/68792.jpg", age, gender, race)
# # # print(result)

# # ModelTesting.model_test_batch("FairFaceMain/fairface_label_test.csv", "FairFaceMain/test", age, gender,race)
# ## test different sizes
# testDiffSizes(age, gender, race, [10,25,50,100,200,500,1000])