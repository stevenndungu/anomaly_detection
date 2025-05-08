import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# import os
# import pandas as pd
# import shutil
# # Define the base directory
# base_dir = "FRGADB_DEST/results/train_test_descriptors/width=150/noperatorspergalaxy=100/DoG"  # Modify this to your actual base directory path

# # Initialize an empty list to store the extracted data
# data = []

# descript_num = 0
# # Loop through each sigma directory
# for sigma_dir in os.listdir(base_dir):
#     sigma_path = os.path.join(base_dir, sigma_dir)
    
#     if os.path.isdir(sigma_path) and sigma_dir.startswith("sigma"):
#         # Extract the sigma value from the directory name (e.g., "sigma=6" => sigma=6)
#         sigma_value = sigma_dir.split("=")[-1]
        
#         # Loop through the subdirectories (like 0-5-25, 0-5-30, etc.)
#         for sub_dir in os.listdir(sigma_path):
#             sub_path = os.path.join(sigma_path, sub_dir)
            
#             if os.path.isdir(sub_path):
#                 # Extract rho from the subdirectory name (e.g., "0-5-25")
#                 rho_value = sub_dir
                
#                 # Loop through the t1 directories (e.g., t1=0.05, t1=0.10)
#                 for t1_dir in os.listdir(sub_path):
#                     t1_path = os.path.join(sub_path, t1_dir)
                    
#                     if os.path.isdir(t1_path) and t1_dir.startswith("t1"):
#                         # Extract t1 value (e.g., "t1=0.05" => t1=0.05)
#                         t1_value = t1_dir.split("=")[-1]
                        
#                         # Loop through the sigma0 directories (e.g., sigma0=0.50, sigma0=0.75)
#                         for sigma0_dir in os.listdir(t1_path):
#                             sigma0_path = os.path.join(t1_path, sigma0_dir)
                            
#                             if os.path.isdir(sigma0_path) and sigma0_dir.startswith("sigma0"):
#                                 # Extract sigma0 value (e.g., "sigma0=0.50" => sigma0=0.50)
#                                 sigma0_value = sigma0_dir.split("=")[-1]
                                
#                                 # Loop through alpha values in the sigma0 directory
#                                 for alpha_dir in os.listdir(sigma0_path):
#                                     alpha_path = os.path.join(sigma0_path, alpha_dir)
                                    
#                                     if os.path.isdir(alpha_path) and alpha_dir.startswith("alpha"):
#                                         # Extract alpha value (e.g., "alpha=0.15" => alpha=0.15)
#                                         alpha_value = alpha_dir.split("=")[-1]

#                                         descript_num += 1
#                                         shutil.copy(f'{alpha_path}/COSFIREdescriptor.mat', f'FRGADB_DEST/descriptors/COSFIREdescriptor_{descript_num}.mat')
                                        
                                        
#                                         # Collect all the extracted values into a dictionary
#                                         data.append({
#                                             "sigma": sigma_value,
#                                             "rho": rho_value,
#                                             "t1": t1_value,
#                                             "sigma0": sigma0_value,
#                                             "alpha": alpha_value,
#                                             "descript_id": descript_num
#                                         })

# # Convert the list of dictionaries to a pandas DataFrame
# df = pd.DataFrame(data)


# print(df)

# # Save the DataFrame to a CSV file if desired
# df.to_csv(f'FRGADB_DEST/descriptors/descriptors.csv', index=False)

dic_labels = {'RRG': 2, 'XRG': 3, 'FRI': 0, 'FRII': 1 }

def get_data(path, dic_labels):
   # Load the MATLAB file
   data = loadmat(path)
   df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
   df1['label'] = 'FRII'
   #df2 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][2])
   #df2['label'] = 'RRG'
   #df3 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][3])
   #df3['label'] = 'XRG'
   #df_train = pd.concat([df0, df1, df2, df3], ignore_index=True)
   df_train = pd.concat([df0, df1], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'RRG'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'XRG'
   df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)
   print(df_train.shape)
   print(df_test.shape)

   column_names = ["descrip_" + str(i) for i in range(1, df_train.shape[1])] + ["label_code"]
   df_train.columns = column_names
   df_test.columns = column_names

   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)

   return df_train, df_test

# Load the dataset
#for i in range(1, 16):

#data, df_test = get_data(path = f'FRGADB_DEST/descriptors/COSFIREdescriptor_{i}.mat', dic_labels = dic_labels)
data, df_test = get_data(path = '/home4/p307791/habrok/projects/experimentation/COSFIRE_Galaxy_Recognition_AD/FRGADB_cleaned_v2/descriptors/COSFIREdescriptor_32.mat', dic_labels = dic_labels)


data.to_csv(f'FRGADB_cleaned_v2/descriptors/COSFIREdescriptor_32_train.csv', index=False)
df_test.to_csv(f'FRGADB_cleaned_v2/descriptors/COSFIREdescriptor_32_test.csv', index=False)

  