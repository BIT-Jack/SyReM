import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 设置字体为Arial
plt.rcParams["font.family"] = ["Arial", "sans-serif"]

# 创建图形和轴，设置尺寸为3x1英寸
fig, ax = plt.subplots(figsize=(7.1, 0.8))

# 隐藏坐标轴
ax.axis('off')

# 创建自定义图例元素
# 红色短线表示observed traj of TV
obs_tv = mlines.Line2D([], [], color='red', marker='', linestyle='-',
                       linewidth=2, label='observed TA traj')

# 蓝色短线表示traj of SV
sv_traj = mlines.Line2D([], [], color='blue', marker='', linestyle='-',
                       linewidth=2, label='observed SA traj')

# 绿色三角形表示ground truth
ground_truth = mlines.Line2D([], [], color='green', marker='^', linestyle='',
                            markersize=8, markerfacecolor='green', label='ground truth')

# 金色五角星表示predicted endpoints
pred_end = mlines.Line2D([], [], color='gold', marker='*', linestyle='',
                        markersize=8, markerfacecolor='gold', label='predicted endpoints')

# 添加图例，调整位置和布局，设置字体大小为8pt
ax.legend(handles=[obs_tv, sv_traj, ground_truth, pred_end],
          loc='center', ncol=4, frameon=False,
          prop={'size': 8})  # 8pt字体

# 紧凑布局，减少留白
plt.tight_layout()

# 保存为SVG文件
plt.savefig('custom_legend_7in.svg', format='svg', bbox_inches='tight', pad_inches=0)

# 显示图像（可选）
plt.show()
    