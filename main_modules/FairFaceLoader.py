## loading in all of the important libraries
import os
import random
import pandas as pd
import numpy as np
import torch as pt
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


## We're going to create a data loader class because this is NEEDED and I am losing it

class FairFaceLoader():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.balanced_df = None
        self.secondary_df = None
        self.sampleSet = None
        self.secondary_sampleSet = None
        self.testSet = None
        
    def __print__(self):
    # function that prints the first 5 rows of the dataframe
        print(self.df.head())
        
    def __len__(self):
    # function that returns the length of the dataframe
        return len(self.df)
    
    def clean(self):
        print(self.df.head())
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
        state_value = np.random.randint(0, 1500) # Use numpy directly to generate a random integer
        self.df = (self.df).sample(frac=1, random_state=state_value).reset_index(drop=True)
    
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
        
    def getBalancedSet(self):
        return self.balanced_df
    
    def getDataSet(self):
        return self.df
    
    def getDecodedLabels(self):
        
        # function that returns the decoded labels
        print(list(zip(self.age_encoder.classes_, range(len(self.age_encoder.classes_)))))
        print(list(zip(self.gender_encoder.classes_, range(len(self.gender_encoder.classes_)))))
        print(list(zip(self.race_encoder.classes_, range(len(self.race_encoder.classes_)))))
        return
    
    def random_int(self):
    # function that generates a random int between two input ranges
        return random.randint(0,1500)
    
    def sampler(self):
        self.secondary_df = self.balance(100)
        # print("Secondary dataframe:")
        # print(self.secondary_df.head())
        self.sampleSet = self.secondary_df.sample(frac=1, random_state=self.random_int()).reset_index(drop=True)
        # print("Sampler set:")
        # print(self.sampleSet.head())
        return self.sampleSet
    
    
    def createTestSet(self, test_size):
        print("Sampler method is not yet implemented.")
        #########################################
        self.secondary_sampleSet = self.sampler().copy()
        source_base_dir = "FairFace" ## where it's coming from
        test_dir = "FairFace/test"   ## where it'll go

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
        
        if len(testFiles) >= 25:
            random_files = random.sample(testFiles, test_size)                  
        else:
            print(f"Warning: Only {len(testFiles)} files available in validation set. Selecting all.")
            
            # Copy the files to the test directory
        for file_relative_path in random_files:
            # source path
            src = os.path.join(source_base_dir, file_relative_path)
            # file name
            file_basename = os.path.basename(file_relative_path)
            # dest pat
            dst = os.path.join(test_dir, file_basename)
            
            # loop through the files and copy them to the test directory
            try:
                if os.path.exists(src):
                    shutil.copy2(src, dst) # Use shutil.copy2 for better copying
                    # print(f"Copied {src} to {dst}")
                else:
                    print(f"Error: Source file {src} does not exist.")
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")

        print("Finished copying files.")
        
        ## Now need to create a new CSV file for all of the information regarding the files in the test directory
        # gen idea is to go call the val csv file and drop all rows that do not have the file name in the test directory

        files = os.listdir(test_dir)
        # Get the file names without the dir path
        files = [os.path.basename(file) for file in files]

        # new of val labels
        ## file column contains file names in the style of val/xxxx.jpg
        # filter df2 to only include rows where the 'file' column matches any of the file names in the test directory
        # Extract the basename from the 'file' column in df2
        
        print("Secondary sample set:")
        print(self.secondary_sampleSet.head())
        
        df2_basenames = self.secondary_sampleSet['file'].str.split('/').str[-1]
        
        print("Basenames:")
        print(df2_basenames.head())

        # Filter df2 where the basename is in the 'files' list
        current_sef_labels_df = self.secondary_sampleSet[df2_basenames.isin(files)].copy() # Use .copy() to avoid some err
        # testertester.drop(columns=['age_label', 'gender_label', 'race_label'], inplace=True)
        # print(testertester)

        ## save as csv in the file path
        current_sef_labels_df.to_csv('FairFace/fairface_label_test.csv', index=False)        
        return random_files
        
        
    

testSet = FairFaceLoader('FairFace/fairface_label_train.csv')
testSet.randomizeSet()
# testSet.balance(100)
# testSet.sampler()
# testSet.clean()
# testSet.balance(100)
# # save as csv
# # print(testSet.balanced_df)
# testSet.randomizeBalancedSet()
testSet.createTestSet(25)
