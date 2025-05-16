#%%
import re
import pandas as pd
import numpy as np
from itables import init_notebook_mode, show
from scipy import stats

# Initialize interactive tables for notebooks
init_notebook_mode(all_interactive=True)

# =============================================================================
# Utility Functions
# =============================================================================

def extract_number(text):
    """
    Extracts the first integer from a string.  
    If no integer is found or if the input is not a string, returns 1.
    """
    if isinstance(text, str):
        matches = re.findall(r'\d+', text)
        return int(matches[0]) if matches else 1
    return 1

def summary_stats_func(df):
    """
    Compute summary statistics from the dataframe description.
    
    For dataframes with multiple columns, it extracts the average of the 
    gmean (geometric mean) across columns. For a single-column dataframe, it 
    extracts the mean and standard deviation.
    
    Returns:
        A tuple (average, std) rounded to two decimals.
    """
    if len(df.columns) > 1:
        # For multiple columns, compute the row-wise description, then transpose
        df_mean = df.describe().iloc[1:2]
        df_mean_T = df_mean.T.reset_index()
        df_mean_T.columns = ['descript_id', 'average_gmean']
        # Use the descriptive stats on the average_gmean column (mean & std)
        stats_desc = df_mean_T.describe().iloc[1:3]
        res = list(stats_desc.average_gmean)
    else:
        # For a single column, extract mean and std directly from describe()
        df_mean = df.describe().iloc[1:3]
        df_mean_T = df_mean.T.reset_index()
        df_mean_T.columns = ['index', 'average_mean', 'average_std']
        res = (df_mean_T['average_mean'].iloc[0], df_mean_T['average_std'].iloc[0])
    
    return np.round(res[0], 2), np.round(res[1], 2)

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

data_path = "model_evaluation_LOF_results_01022025_without_outliers_train_FRGADB_cleaned_v2_valid_test_normalized_n_filters5_to101_by1.csv"
df = pd.read_csv(data_path)

# Filter data to only include rows with 50 filters
df = df.query('num_filters in [40]')

# =============================================================================
# Determine Significant Hyperparameters from Validation Data
# =============================================================================

# Select only columns starting with 'gmean_score_valid'
df_valid = df.loc[:, df.columns.str.startswith('gmean_score_valid')].copy()

# Compute the average gmean score per row and add it as a new column
df_valid['average_gmean_score_valid'] = df_valid.mean(axis=1)

# Sort the dataframe in descending order by the average gmean score
df_valid.sort_values(by='average_gmean_score_valid', ascending=False, inplace=True)

# Identify the row with the maximum average gmean score
max_index = df_valid['average_gmean_score_valid'].idxmax()
max_row = df_valid.loc[max_index]

# Perform independent t-tests comparing the max_row to each row in df_valid
pvalues_t = []
pvalues_real_t = []

for idx in df_valid.index:
    # Exclude the average column from the t-test using slicing ([:-1])
    _, p_value = stats.ttest_ind(max_row[:-1], df_valid.loc[idx][:-1], alternative='greater')
    pvalues_real_t.append(p_value)
    pvalues_t.append(1 if p_value >= 0.05 else 0)

# Create a copy to hold the t-test results
df_valid_t = df_valid.copy()
df_valid_t['pvalues_t'] = pvalues_real_t
df_valid_t['sig_t'] = pvalues_t

# Filter rows where the t-test indicates significance (sig_t == 1)
df_valid_sig = df_valid_t.query('sig_t == 1')
# (Optional) Identify the column with the maximum mean across significant results
max_indexx = df_valid_sig.iloc[:, :-3].mean().idxmax()

# =============================================================================
# Evaluate Hyperparameters on Test Data
# =============================================================================

# ---- Validation Data Summary ----

# Compute descriptive statistics for the significant validation hyperparameters
valid_summary = df_valid_sig.describe().iloc[1:2]
# Optionally, restrict to a subset of columns (here, the first 32)
valid_summary = valid_summary[df_valid_sig.columns[:32]]
valid_summary_T = valid_summary.T
valid_summary_T.sort_values(by='mean', ascending=False, inplace=True)

# Extract the best descriptor from validation data
best_descriptor_df = valid_summary_T.reset_index()
best_descriptor_df.columns = ['descript_id', 'average_gmean']
best_descriptor_name = best_descriptor_df['descript_id'].iloc[0]
best_gmean_average = best_descriptor_df['average_gmean'].iloc[0]

# Uncomment these lines to print the best descriptor info:
# print(f'Best Valid descriptor: {best_descriptor_name}')
# print(f'Average Gmean valid: {np.round(best_gmean_average * 100, 2)} %')

# Detailed summary for valid gmean scores (optional display)
valid_detail_summary = df_valid_sig.loc[:, df_valid_sig.columns.str.startswith('gmean_score_valid')].describe().iloc[1:3]
show(valid_detail_summary)

# ---- Test Data Summary ----

# Select rows from the original dataframe corresponding to significant validation indices
optimal_params = df.loc[df_valid_sig.index].copy()
optimal_params.reset_index(drop=True, inplace=True)

# Select columns starting with 'gmean_score_test' from the optimal parameters
test_results = optimal_params.loc[:, optimal_params.columns.str.startswith('gmean_score_test')]

# Create a summary for test results
test_results_summary = (
    test_results.describe().iloc[1:3]
    .T
    .reset_index()
)
test_results_summary.columns = ['descript_id', 'gmean', 'std']
test_results_summary['id'] = test_results_summary['descript_id'].apply(extract_number)
test_results_summary.sort_values(by='gmean', inplace=True)
# Optionally save the summary:
# test_results_summary.to_csv('test_res_summary.csv', index=False)

# =============================================================================
# Mann-Whitney U Tests for Additional Significance Check
# =============================================================================

pvalues_mw = []
pvalues_real_mw = []
sig_descriptor = pd.DataFrame()

# Compare each valid descriptor with the best descriptor using a one-sided Mann-Whitney U test
for col in df_valid_sig.columns[df_valid_sig.columns.str.startswith('gmean_score_valid')]:
    _, p_value_mw = stats.mannwhitneyu(
        df_valid_sig[best_descriptor_name],
        df_valid_sig[col],
        alternative='greater'
    )
    pvalues_real_mw.append(p_value_mw)
    pvalues_mw.append(1 if p_value_mw >= 0.05 else 0)
    if p_value_mw >= 0.05:
        sig_descriptor[col] = df_valid_sig[col]

# Construct the list of significant test results columns based on valid descriptors
significant_results = [f'gmean_score_test_{extract_number(col)}' for col in sig_descriptor.columns]

print(f"Number of significant descriptors: {len(significant_results)}")
print("Significant descriptors:", significant_results)

# Extract the test results for significant descriptors and compute summary statistics
df_significant_descriptors = optimal_params[significant_results]
result_summary = summary_stats_func(df_significant_descriptors)
print("Summary stats (mean, std) for significant descriptors on test data:", result_summary)

# %%
