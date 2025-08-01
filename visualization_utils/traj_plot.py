import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import random
from visualization_utils import map_vis_without_lanelet
import math
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm


def rotation_matrix(rad):
    psi = rad - math.pi/2
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])


def get_polygon_cars(center, width, length, radian):
    x0, y0 = center[0], center[1]
#     lowleft = (x0 - length / 2., y0 - width / 2.)
#     lowright = (x0 + length / 2., y0 - width / 2.)
#     upright = (x0 + length / 2., y0 + width / 2.)
#     upleft = (x0 - length / 2., y0 + width / 2.)
    lowleft = (- length / 2., - width / 2.)
    lowright = ( + length / 2., - width / 2.)
    upright = ( + length / 2., + width / 2.)
    upleft = ( - length / 2., + width / 2.)
    rotate_ = rotation_matrix(radian)
    
    return (np.array([lowleft, lowright, upright, upleft])).dot(rotate_)+center

# Function definitions (rotation_matrix, get_polygon_cars) remain the same

#=================settings==================


# cl_methods = ['vanilla', 'vanilla', 'vanilla', 'vanilla', 'b2p', 'b2p', 'b2p', 'b2p']
cl_methods = ['vanilla', 'vanilla', 'vanilla', 'vanilla', 'agem_r', 'agem_r', 'agem_r', 'agem_r']
# cl_methods = ['agem_r', 'agem_r', 'agem_r', 'agem_r', 'agem_r_pe', 'agem_r_pe', 'agem_r_pe', 'agem_r_pe']
buffers = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
learned_tasks = [2, 3, 4, 5, 2, 3, 4, 5]  # Assuming all are the same, you can modify each entry if needed
test_sce = 'FT'
mapfile = './mapfiles/DR_USA_Roundabout_FT.osm'
# test_sce = 'MA'
# mapfile = './mapfiles/DR_USA_Intersection_MA.osm'
#================loading====================
data_prediction = np.load('./logging/prediction_record/'+cl_methods[0]+'_buffer'+str(buffers[0])+'_'+str(learned_tasks[0])+'tasks_learned_test_on_'+test_sce+'.npz')
data_error = np.load('./logging/prediction_record/fde_mr_'+cl_methods[0]+'_buffer'+str(buffers[0])+'_'+str(learned_tasks[1])+'tasks_learned_test_on_'+test_sce+'.npz')
data_obs_pose = np.load('./cl_dataset/val/val_'+test_sce+'.npz')
raw_original_tv_information = np.load('./logging/original_reference/'+test_sce+'_target.npz')

case_fde = data_error['all_case_fde']
case_mr = data_error['all_case_mr'] 

#==================================plotting====================================================

case_id = np.argmax(case_mr)
# case_id = np.argmin(case_mr)
# case_id = random.randint(0, len(data_prediction['pred']))
# case_id = np.argmax(case_fde)
# case_id = np.argmin(case_fde)
# case_id = 536



print("case id:", case_id)


# Set font to Arial and size 9
plt.rc('font', family='Arial', size=9)

fig = plt.figure(figsize=(7.16, 4.4))
gs = gridspec.GridSpec(2, 4, figure=fig, left=0.048, right=0.98, top=0.95, bottom=0.01, wspace=0.04, hspace=0)

# Predefine handles for the legend
handles = []

for i in range(8):
    cl_method = cl_methods[i]
    buffer = buffers[i]
    learned_task = learned_tasks[i]
    
    # Reload data if different cl_method or buffer is needed
    if i > 0:
        data_prediction = np.load('./logging/prediction_record/'+cl_method+'_buffer'+str(buffer)+'_'+str(learned_task)+'tasks_learned_test_on_'+test_sce+'.npz')
        data_error = np.load('./logging/prediction_record/fde_mr_'+cl_method+'_buffer'+str(buffer)+'_'+str(learned_task)+'tasks_learned_test_on_'+test_sce+'.npz')
        data_obs_pose = np.load('./cl_dataset/val/val_'+test_sce+'.npz')
        raw_original_tv_information = np.load('./logging/original_reference/'+test_sce+'_target.npz')
    
    # Observations:
    obs_traj_all_agents = data_obs_pose['trajectory'][case_id] 
    obs_pose_all_agents = data_obs_pose['pose_shape'][case_id] 
    tv_traj_x = obs_traj_all_agents[0][:,2] 
    tv_traj_y = obs_traj_all_agents[0][:,3]
    tv_origin_x = obs_traj_all_agents[0][-1,2]
    tv_origin_y = obs_traj_all_agents[0][-1,2]
    raw_tv_origin_x = raw_original_tv_information['x'][case_id]
    raw_tv_origin_y = raw_original_tv_information['y'][case_id]
    tv_heading = obs_pose_all_agents[0][-1,0]
    tv_width = obs_pose_all_agents[0][-1,1]
    tv_length = obs_pose_all_agents[0][-1,2]

    tv_vx = obs_traj_all_agents[0][8,4]
    tv_vy = obs_traj_all_agents[0][8,5]
    tv_v = (tv_vx**2+tv_vy**2)**0.5

    # Predictions:
    prediction = data_prediction['pred'][case_id]
    ground_truth = data_prediction['gt'][case_id]
    pred_heatmap = data_prediction['heatmap'][case_id]

    x_gt = ground_truth[0]
    y_gt = ground_truth[1]
    x_pred = prediction[:, 0]
    y_pred = prediction[:, 1]

    x = np.arange(-22.75, 23.25, 0.5)
    y = np.arange(-11.75, 75.25, 0.5)
    s = pred_heatmap/np.amax(pred_heatmap)
    s[s<0.006]=np.nan

    raw_origin = (raw_tv_origin_x, raw_tv_origin_y)
    origin = (tv_origin_x, tv_origin_y)
    rotate = rotation_matrix(tv_heading)
    xrange = [-25, 25]
    yrange = [-10, 35]

    # Plot in the appropriate subplot
    ax = fig.add_subplot(gs[i])
    
    # maps
    map_vis_without_lanelet.draw_map_without_lanelet(mapfile, ax, raw_origin, rotate, xrange, yrange)
    ax.pcolormesh(x, y, s.transpose(), cmap='Reds', zorder=0)

    bbox_tv = get_polygon_cars(origin, tv_width, tv_length, 0)
    rect_tv = matplotlib.patches.Polygon(bbox_tv, closed=True, facecolor='r', edgecolor='r', linewidth=1, alpha=0.5, zorder=20)
    ax.add_patch(rect_tv)

    for ii in range(1, 26):
        sv_traj_x = obs_traj_all_agents[ii][:,2]
        sv_traj_y = obs_traj_all_agents[ii][:,3]
        if np.all(sv_traj_x == 0) and np.all(sv_traj_y == 0): 
            continue
        elif sv_traj_x[0] == 0: 
            sv_traj_x = obs_traj_all_agents[ii][1:,2]
            sv_traj_y = obs_traj_all_agents[ii][1:,3]
        ax.plot(sv_traj_x, sv_traj_y, color='b')
        sv_current_x = obs_traj_all_agents[ii][8, 2]
        sv_current_y = obs_traj_all_agents[ii][8, 3]
        sv_current_head_relative = -(obs_pose_all_agents[ii][8, 0] - tv_heading)
        sv_width = obs_pose_all_agents[ii][8, 1]
        sv_length = obs_pose_all_agents[ii][8, 2]
        if sv_length == 0 and sv_width == 0:
            h_sv = ax.scatter(sv_current_x, sv_current_y, marker='o', color='b', label='SV')
        else:
            bbox_sv = get_polygon_cars((sv_current_x, sv_current_y), sv_width, sv_length, sv_current_head_relative)
            rect_sv = matplotlib.patches.Polygon(bbox_sv, closed=True, facecolor='b', edgecolor='b', linewidth=1, alpha=0.5, zorder=20)
            ax.add_patch(rect_sv)
            h_sv, = ax.plot([], [], color='b', label='observed trajectories of SV')  # Dummy for legend

    # Other plotting code...

    h_tv, = ax.plot(tv_traj_x, tv_traj_y, label='observed trajectories of TV', color='r')
    h_pred = ax.scatter(x_pred, y_pred, marker='*', s=8, color='gold', label='predicted endpoint', zorder=200)
    h_gt = ax.scatter(x_gt, y_gt, marker='^', s=8, color='green', label='ground truth', zorder=199)

    # Set axis ticks and their properties
    if i not in [0, 4]:  # Hide y ticks for all but the first column
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    if i in [0,1,2,3]:
        if i==0:
            ax.set_title('Vanilla: after learning '+str(learned_task)+' tasks', fontsize=9, pad=2)
        else:
            ax.set_title('Vanilla: after learning '+str(learned_task)+' tasks', fontsize=9, pad=2)
    else:
        if i==4:
            ax.set_title('Ours: after learing '+str(learned_task)+' tasks', fontsize=9, pad=2)
        else:
            ax.set_title('Ours: after learing '+str(learned_task)+' tasks', fontsize=9, pad=2)

    # ax.set_xticklabels([])
    ax.set_xticks([-20, -10, 0, 10, 20])
    # if i==1 or i==5:
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.tick_params(axis='both', which='major', width=0.5, labelsize=8)

    # Collect handles for the legend
    # if i == 0:

    #     handles = [h_tv, h_sv, h_gt, h_pred]


# Add the unified legend
# fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9, frameon=False)

plt.tight_layout()
# plt.savefig('./traj_vis_4tasks_V3.pdf')
plt.show()
