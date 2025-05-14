
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

sns.set_style("white")
SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

df = pd.read_csv('test_res_summary.csv')
df = df[['gmean', 'std']]
df['descriptor_id_new'] = np.array(range(32)) + 1

fig, ax = plt.subplots(figsize=(10/3, 3))
sns.lineplot(data=df, x='descriptor_id_new', y='gmean',  marker='o')
plt.fill_between(df['descriptor_id_new'], df['gmean'] - df['std'], df['gmean'] + df['std'], color='#439CEF')
#sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  marker='D')
#sns.lineplot(data=df_gcnn, x='topk_number_images', y='mAP',label='G-CNNs',  marker='s')
# plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

plt.xlabel('COSFIRE descriptors')
plt.ylabel('Geometric Mean')
plt.ylim(65, 94)
plt.savefig(f'test_results_descriptors.png')
plt.savefig(f'test_results_descriptors.svg',format='svg', dpi=1200)
plt.show()

# %%
###########


df_cosfire =  pd.read_csv('map_vs_topK_images_IR/mAP_vs_topk_images_72bit_cosfire_mean_std.csv')
df_densenet =  pd.read_csv('map_vs_topK_images_IR/mAP_vs_215_images_72bit_densenet_mean_std.csv')
df_gcnn =  pd.read_csv('map_vs_topK_images_IR/mAP_vs_215_images_72bit_gcnn_v2.csv')

fig, ax = plt.subplots(figsize=(10/3, 3))
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('font', family='Nimbus Roman')
sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='COSFIRE',  marker='o')
# plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet-161',  marker='D')
sns.lineplot(data=df_gcnn, x='topk_number_images', y='mAP',label='G-CNNs',  marker='s')
# plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP (%)')
plt.ylim(70, 95)
plt.savefig(f'map_vs_topK_images_IR/map_vs_topk_number_images_gcnns_cbd_v2.png')
plt.savefig(f'map_vs_topK_images_IR/map_vs_topk_number_images_gcnns_cbd.svg',format='svg', dpi=1200)
plt.show()

# %%
# GCNN average mAP when R=K:  89.69