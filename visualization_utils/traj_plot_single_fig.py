import numpy as np
from visualization_utils import map_vis_without_lanelet
import math
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms
import random

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

#=================settings==================
# cl_method = 'agem_r_pe'
# cl_method = 'agem_r'

cl_method = 'agem_r'
buffer = 1000
# buffer = 0
learned_tasks = 1
test_sce = 'MA'
mapfile = './mapfiles/DR_USA_Intersection_MA.osm'
# mapfile = './mapfiles/DR_USA_Roundabout_FT.osm'
# mapfile = './mapfiles/DR_CHN_Roundabout_LN.osm'
# mapfile = './mapfiles/DR_CHN_Merging_ZS2.osm'
# mapfile = './mapfiles/DR_DEU_Roundabout_OF.osm'
# mapfile = './mapfiles/DR_USA_Intersection_EP0.osm'
#================loading====================
data_prediction = np.load('./logging/prediction_record/'+cl_method+'_buffer'+str(buffer)+'_'+str(learned_tasks)+'tasks_learned_test_on_'+test_sce+'.npz')
data_error = np.load('./logging/prediction_record/fde_mr_'+cl_method+'_buffer'+str(buffer)+'_'+str(learned_tasks)+'tasks_learned_test_on_'+test_sce+'.npz')
data_obs_pose = np.load('./cl_dataset/val/val_'+test_sce+'.npz')
raw_original_tv_information = np.load('./logging/original_reference/'+test_sce+'_target.npz')

case_fde = data_error['all_case_fde']
case_mr = data_error['all_case_mr'] 

#/////////////////////////choose the case to show//////////////////////////////////////
# case_id = np.argmax(case_mr)
# case_id = np.argmin(case_mr)
# case_id = random.randint(0, len(data_prediction['pred']))
# print("case id:", case_id)
# case_id = np.argmax(case_fde)
# case_id = np.argmin(case_fde)
case_id = 129
#/////////////////////////////////////////////////////////////////////////////////////
print("case id:", case_id)


# Observations:
obs_traj_all_agents = data_obs_pose['trajectory'][case_id] #historical trajectories (including features of postion and velocity)
obs_pose_all_agents = data_obs_pose['pose_shape'][case_id] # width, length, headings
## target vehicles
tv_traj_x = obs_traj_all_agents[0][:,2] 
tv_traj_y = obs_traj_all_agents[0][:,3]
tv_origin_x = obs_traj_all_agents[0][-1,2] # 0
tv_origin_y = obs_traj_all_agents[0][-1,2] # 0
raw_tv_origin_x = raw_original_tv_information['x'][case_id] # for map coordinates transferring 
raw_tv_origin_y = raw_original_tv_information['y'][case_id]


tv_heading = obs_pose_all_agents[0][-1,0] #heading angel(rad)
tv_width = obs_pose_all_agents[0][-1,1]
tv_length = obs_pose_all_agents[0][-1,2]



tv_vx = obs_traj_all_agents[0][8,4]
tv_vy = obs_traj_all_agents[0][8,5]
tv_v = (tv_vx**2+tv_vy**2)**0.5

# if tv_v>0 and (math.cos(tv_heading)*tv_vx+math.sin(tv_heading)*tv_vy<0):
#     tv_heading = tv_heading+math.pi
## information of surrounding vehicels will be unpacked with "for loops" in plot part



# Predictions:
prediction = data_prediction['pred'][case_id]
ground_truth = data_prediction['gt'][case_id]
pred_heatmap = data_prediction['heatmap'][case_id]

x_gt = ground_truth[0]
y_gt = ground_truth[1]
x_pred = prediction[:, 0]
y_pred = prediction[:, 1]


print("original data loaded.")



# Field of Vision
x = np.arange(-22.75, 23.25, 0.5)
y = np.arange(-11.75, 75.25, 0.5)
s = pred_heatmap/np.amax(pred_heatmap)
s[s<0.006]=np.nan



raw_origin = (raw_tv_origin_x, raw_tv_origin_y) # the original data extracted from csv files, for map coordinate transfer
origin = (tv_origin_x, tv_origin_y) # in the coordinate sys of target vehicles (0,0)
rotate = rotation_matrix(tv_heading) # ratation matrix
xrange = [-30, 15]
yrange = [-10, 30]





#==================================plotting====================================================
fig, axes = plt.subplots(1, 1, figsize=(3,2.5))
# maps
map_vis_without_lanelet.draw_map_without_lanelet(mapfile, axes, raw_origin, rotate, xrange, yrange)
axes.pcolormesh(x, y, s.transpose(), cmap='Reds', zorder=0)

# rectangles to represent vehicles
bbox_tv = get_polygon_cars(origin, tv_width, tv_length, 0)
rect_tv = matplotlib.patches.Polygon(bbox_tv, closed=True, facecolor='red', edgecolor='red', linewidth=1, alpha=0.5, zorder= 20)
axes.add_patch(rect_tv)

# information of surronding vehicles

for ii in range(1, 26):
    sv_traj_x = obs_traj_all_agents[ii][:,2]
    sv_traj_y = obs_traj_all_agents[ii][:,3]
    if np.all(sv_traj_x==0) and np.all(sv_traj_y==0): # no agent
        continue
    elif sv_traj_x[0] == 0: # for which without full observation time
        sv_traj_x = obs_traj_all_agents[ii][1:,2]
        sv_traj_y = obs_traj_all_agents[ii][1:,3]
    if ii==1:
        plt.plot(sv_traj_x, sv_traj_y, label = 'SV', color='b')
    else:
        plt.plot(sv_traj_x, sv_traj_y, color='b')
    sv_current_x = obs_traj_all_agents[ii][8, 2]
    sv_current_y = obs_traj_all_agents[ii][8, 3]
    # sv_current_head_relative = (obs_pose_all_agents[ii][8, 0]-tv_heading)
    sv_current_head_relative = -(obs_pose_all_agents[ii][8, 0]-tv_heading)
    sv_width = obs_pose_all_agents[ii][8, 1]
    sv_length = obs_pose_all_agents[ii][8, 2]
    if sv_length == 0 and sv_width ==0:
        plt.scatter(sv_current_x, sv_current_y, marker='o', color='pink', label='pedestrian/cyclist')
    else:
        bbox_sv = get_polygon_cars((sv_current_x, sv_current_y), sv_width, sv_length, sv_current_head_relative)
        rect_sv = matplotlib.patches.Polygon(bbox_sv, closed=True, facecolor='blue', edgecolor='blue', linewidth=1, alpha=0.5, zorder=20)
        axes.add_patch(rect_sv)



plt.plot(tv_traj_x, tv_traj_y, label='TV', color='r')
plt.scatter(x_pred, y_pred, marker='*', color='gold', label='predicted', zorder = 200)
plt.scatter(x_gt, y_gt, marker='^', color='green', label='ground truth', zorder = 199)
# plt.legend(loc='upper left')

fig.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.98)
plt.savefig(f'./case study figures/{cl_method}_learned'+str(learned_tasks)+'_test_in_'+test_sce+'_case'+str(case_id)+'.svg')
# plt.savefig(f'./case study figures/network_schematic/map.svg')
plt.show()