#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor  
from sklearn.model_selection import train_test_split
from utils import *
from functools import reduce

experiment_results = []
experiment_results_test = []
merge_results = []
merge_results_test = []

dic_labels = {'RRG': 2, 'XRG': 3, 'FRI': 0, 'FRII': 1}
# Define hyperparameters
n_neighbors = [8,10] 
contamination = ["auto",0.37,0.38]
leaf_sizes = [40,60,80]
algorithms = ["auto"]
metric = ["cosine"]
temp_df = pd.DataFrame(columns=['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size'])
#%%
# Loop through the datasets
for i in range(1, 33):

   # Load the dataset
   path = f'FRGADB_cleaned/descriptors/COSFIREdescriptor_{i}.mat'
   data, df_test = get_data(path, dic_labels)
   class_0_1 = data[data['label_code'].isin([0, 1])]
   class_2_3 = data[data['label_code'].isin([2, 3])]
   train_0_1, valid_0_1 = train_test_split(class_0_1, test_size=0.1, random_state=42, stratify=class_0_1['label_code'])
   train_2_3, valid_2_3 = train_test_split(class_2_3, test_size=0.5, random_state=42, stratify=class_2_3['label_code'])
   train_data = pd.concat([train_0_1, train_2_3])
   valid_data = pd.concat([valid_0_1, valid_2_3])
   



   # Train data
   train_data = train_data.query('label_code in [0,1]')     
   print('Distribution of the labels in the training set')
   print(train_data.label_code.value_counts())
   X_train = train_data.drop(columns=['label_code'])
   X_train = preprocessing.normalize(X_train.values)
   y_train = train_data['label_code']

   # Valid data
   print('Distribution of the labels in the Valid set')
   print(valid_data.label_code.value_counts())
   X_valid = valid_data.drop(columns=['label_code'])
   X_valid = preprocessing.normalize(X_valid.values)
   y_valid = valid_data['label_code'].apply(lambda x: 0 if x in [0, 1] else 1)

   # Test data
   X_test = df_test.drop(columns=['label_code'])
   X_test = preprocessing.normalize(X_test.values)
   y_test = df_test['label_code']
   y_test = df_test['label_code'].apply(lambda x: 0 if x in [0, 1] else 1)
   print('Distribution of the labels in the Test set')
   print(df_test.label_code.value_counts())
   
   


   # Loop through the hyperparameters
   for n_neigh in n_neighbors:
       for cont in contamination:
               for algorithm in algorithms:
                     for leaf_size in leaf_sizes:
                        for met in metric:  # Add metric to the loop
                           # Initialize LocalOutlierFactor
                           clf = LocalOutlierFactor(n_neighbors=n_neigh,
                                                      contamination=cont,
                                                      metric=met, 
                                                      leaf_size = leaf_size, 
                                                      algorithm=algorithm,
                                                      novelty=True)  # novelty=True for outlier detection

                           # Fit the model                           
                           clf.fit(X_train)                                                     
                           final_predictions = clf.predict(X_valid)
                           final_predictions = np.where(final_predictions == -1, 1, 0)
                           valid_results = evaluate_model(y_valid, final_predictions)

                           final_predictions_test = clf.predict(X_test)
                           final_predictions_test = np.where(final_predictions_test == -1, 1, 0)
                           test_results = evaluate_model(y_test, final_predictions_test)

                           results = {                               
                              'n_neighbors': n_neigh,
                              'contamination': cont,                           
                              'algorithm': algorithm,
                              'metric': met ,
                              "leaf_size": leaf_size,
                              'accuracy': valid_results[0],
                              'precision': valid_results[1], 
                              'recall': valid_results[2], 
                              'f1_score': valid_results[3], 
                              'specificity': valid_results[4], 
                              'f2_score': valid_results[5], 
                              'gmean_score': valid_results[6]
                           }

                           results_test = {                               
                              'n_neighbors': n_neigh,
                              'contamination': cont,                           
                              'algorithm': algorithm,
                              'metric': met ,
                              "leaf_size": leaf_size,
                              'accuracy': test_results[0],
                              'precision': test_results[1], 
                              'recall': test_results[2], 
                              'f1_score': test_results[3], 
                              'specificity': test_results[4], 
                              'f2_score': test_results[5], 
                              'gmean_score': test_results[6]
                           }

                           # Valid results
                           df = pd.DataFrame([results])                           
                           df['descriptor_id'] = str(i)
                           experiment_results.append(df)
                           combined_results = pd.concat(experiment_results, ignore_index=True)
                           combined_results.sort_values(by=['gmean_score'], inplace=True, ascending=False)
                           combined_results.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_valid.csv', index=False)
                           combined_results2= combined_results[['descriptor_id','f1_score','gmean_score']]
                           combined_results2.to_csv('model_evaluation_LOF_results_21012025_v2_without_outliers_train_FRGADB_cleaned_valid.csv', index=False)
                           print(combined_results2.head(5))

                           # Test results
                           df_testing = pd.DataFrame([results_test])                           
                           df_testing['descriptor_id'] = str(i)
                           experiment_results_test.append(df_testing)
                           combined_results_test = pd.concat(experiment_results_test, ignore_index=True)
                           combined_results_test.sort_values(by=['gmean_score'], inplace=True, ascending=False)
                           combined_results_test.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_test.csv', index=False)
                           combined_results2_test= combined_results_test[['descriptor_id','f1_score','gmean_score']]
                           combined_results2_test.to_csv('model_evaluation_LOF_results_21012025_v2_without_outliers_train_FRGADB_cleaned_test.csv', index=False)
                           print(combined_results2_test.head(5))

                           
                           
   combined_results = combined_results[['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size','descriptor_id', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'f2_score', 'gmean_score']]
   combined_results.columns = ['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size',f'descriptor_id_{i}', f'accuracy_valid_{i}', f'precision_valid_{i}', f'recall_valid_{i}', f'f1_score_valid_valid_{i}', f'specificity_valid_{i}',
       f'f2_score_valid_{i}', f'gmean_score_valid_{i}']   
   print(combined_results.shape)   
   merge_results.append(combined_results)
   experiment_results = []

   combined_results_test = combined_results_test[['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size','descriptor_id', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'f2_score', 'gmean_score']]
   combined_results_test.columns = ['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size',f'descriptor_test_id_{i}', f'accuracy_test_{i}', f'precision_test_{i}', f'recall_test_{i}', f'f1_score_test_{i}', f'specificity_test_{i}',
       f'f2_score_test_{i}', f'gmean_score_test_{i}']   
   #print(combined_results_test.shape)   
   merge_results_test.append(combined_results_test)
   experiment_results_test = []
        
  
   
   
   df2 = reduce(lambda x, y: x.merge(y, on=['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size']), merge_results) 
   df2.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_valid.csv', index=False)

   df_testing2 = reduce(lambda x, y: x.merge(y, on=['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size']), merge_results_test) 
   df_testing2.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_test.csv', index=False)

   valid_test_df = reduce(lambda x, y: x.merge(y, on=['n_neighbors', 'contamination', 'algorithm', 'metric', 'leaf_size']), [df2, df_testing2]) 
   valid_test_df.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_valid_test.csv', index=False)

   print('Completed descriptor id: ', i)

# valid_test_df_v2 = valid_test_df.sort_values(by=['gmean_score'], inplace=True, ascending=False)
# valid_test_df_v2.to_csv('model_evaluation_LOF_results_21012025_without_outliers_train_FRGADB_cleaned_valid_test_v2.csv', index=False)
print('Complete ......')
   # %%
