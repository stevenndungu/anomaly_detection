#  Anomaly Detection COSFIRE Algorithm [Guide]

This is the official repository for the paper 'Anomaly detection in radio galaxy data with trainable COSFIRE filters'

Download the complete **[COSFIRE Algorithm code](https://zenodo.org/me/uploads)** 

1. Navigate to the cosfire_rg_classification > Application directory and set it as your current working directory.

2. Load MATLAB in the terminal:
   - For an HPC environment: `module load MATLAB/2022b-r5`
  

3. Execute the following command to generate scripts for running the different hyperparameter sets (32 in this case). This script helps you create individual scripts to be executed:

4. Run the individual scripts to obtain the descriptors, for instance,
- When running locally execute:
  ```
  Galaxy_recognition_with_COSFIRE(100,'/scratch/p307791/radio/',150,5.00,0:5:20,0.05,0.50,0.15,1,10,1000)
  ```
- When running on HPC execute:
  ```
  sbatch scripts/script1.sh
  ```

5. In order to obtain the train, validation, and test descriptors, execute the [extract_descriptors.py](./processing_scripts/extract_descriptors.py) script which extracts the descriptors from the results folder (where all the results of the executed runs are stored). Refer to [Galaxy_recognition_with_COSFIRE.m](./Application/Galaxy_recognition_with_COSFIRE.m) script for details.

6. To obtain the final results, execute the [anomaly_detection_evaluation.qmd](./anomaly_detection_evaluation.qmd) or refer to **[Anomaly detection GitHub Page](https://stevenndungu.github.io/anomaly_detection/)** where code and explanations are embedded.

Note:

- The script to generate synthetic anomalies:  **[synthetic_images_creation.py](./processing_scripts/synthetic_images_creation.py)** 

- We have provided the descriptors for the training, validation, and test sets in the **[descriptors](./descriptors)** folder. You can use them directly without running the COSFIRE algorithm again.

- The final results are stored in the **[results](./results)** folder. The data is processed and analysed in the **[anomaly_detection_evaluation.qmd](./anomaly_detection_evaluation.qmd)** file. Refer to the refer to **[Anomaly detection GitHub Page](https://stevenndungu.github.io/anomaly_detection/)** for more details.

- To determine the best number of COSFIRE filters, refer to the **[optimal_number_of_COSFIRE_filters_per_class_analyses.py](./processing_scripts/optimal_number_of_COSFIRE_filters_per_class_analyses.py)** script. 