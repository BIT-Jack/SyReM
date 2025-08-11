import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 用户自定义文件名

# model_name = 'agem_r_pe'
model_name = 'agem_new'

fontsize = 12
file_path = f'./connected/{model_name}_bf_1000_STEP_SCORE_total.pkl'

# 加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

downsampling_rate = 100
data = data[::downsampling_rate]  # Downsample the data


num_steps = len(data)
steps = list(range(1, num_steps+1))

batch_size = 8
scores_matrix = np.zeros((batch_size, num_steps))

for j, batch in enumerate(data):
    if 'new' in model_name:
        batch.sort(reverse=True)
    for sample_id in range(0,len(batch)):
        scores_matrix[sample_id, j] = batch[sample_id]

print(scores_matrix.shape)

# 设置绘图
plt.figure(figsize=(14.32, 3.6))  # 固定的图像大小
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})

# # 使用 seaborn 的 heatmap 绘制统计矩阵
# if 'der' in model_name:
#     cmap = sns.color_palette("Reds", as_cmap=True)  # 使用红色渐变色调
# else:
#     cmap = sns.color_palette("Blues", as_cmap=True)  # 使用蓝色渐变色调
color_set = 'RdBu'
cmap = sns.color_palette(color_set, as_cmap=True)

vmin = -1
vmax = 1

# 绘制 heatmap
ax = sns.heatmap(scores_matrix, cmap=cmap, cbar_kws={ 'pad': 0.01}, annot=False, fmt="d", 
                 linewidths=0.1, linecolor='white', vmin=vmin, vmax=vmax)  # General linecolor for the grid

# 设置colorbar 文字大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Similarity Score', fontsize=fontsize)



# # 设置网格线透明度
# for _, spine in ax.spines.items():  # Iterate over the grid line (spines)
#     spine.set_visible(True)
#     spine.set_alpha(0.1)  # Set transparency for grid lines (0 = fully transparent, 1 = fully opaque)

# Ensure spines are visible and adjust the color and linewidth
for spine in ax.spines.values():
    spine.set_visible(True)  # Make sure the spines are visible
    spine.set_linewidth(1.5)  # Set linewidth to 1.5
    spine.set_color('black')  # Set the color of the spines to black
    # set the font size of color bar
    


#设置轴标签
if 'new' in model_name:
    plt.title('Rehearsed Memory of SyReM-R', fontsize=fontsize)
else:
    plt.title('Rehearsed Memory of SyReM', fontsize=fontsize)

plt.xlabel('Learning Step', fontsize = fontsize)
plt.ylabel('Index of Sample in Each Batch', fontsize = fontsize)
plt.gca().invert_yaxis()  # Invert Y-axis to have Task ID 1 at the top

# 设置 Y 轴刻度
ax.set_yticks(np.arange(8) + 0.5)  # 将标签放置在格子的中心
ax.set_yticklabels(np.arange(1, 9), fontsize=fontsize, rotation=0)  

# 设置 X 轴刻度
# Original indices: multiply downsampled index by downsampling_rate
original_x_ticks = np.arange(1, num_steps + 1,25) * downsampling_rate  # Scale by downsampling rate
ax.set_xticks(np.arange(1, num_steps + 1,25) + 0.5)  # Set tick positions in the center of each cell
ax.set_xticklabels(original_x_ticks, fontsize=fontsize, rotation=0)  # Set the tick labels to the original indices

# 显示图像
plt.tight_layout()
# plt.show()
# save the fig as PDF
plt.savefig(f'./outputs/{model_name}_sample_scores_{color_set}.svg', bbox_inches='tight')
