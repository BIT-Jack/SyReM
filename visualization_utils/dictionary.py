import pickle
#This code serves running extract_original_tv_info.py
#The input of this code is "val_index.pickle" which stores the track ids of target vehicles for each case among all datasets

def arrays_to_dict(keys, values):
    return {k: v for k, v in zip(keys, values)}



with open('./logging/original_reference/val_index.pickle', 'rb') as handle:
    s = pickle.load(handle)
# print(len(s))
caselist, carid = s[0], s[1]

# case_id_ZS0 = caselist[0:1084]
case_id_ZS0 = [idx for idx in range(1, 1085, 1)]
target_veh_id_ZS0 = carid[0:1084]

# case_id_ZS2 = caselist[1084:2127]
case_id_ZS2 = [idx for idx in range(1, 1044, 1)]
target_veh_id_ZS2 = carid[1084:2127]

# case_id_LN = caselist[2127:2365]
case_id_LN = [idx for idx in range(1, 239, 1)]
target_veh_id_LN = carid[2127:2365]

# case_id_MT = caselist[2365:2708]
case_id_MT = [idx for idx in range(1, 344, 1)]
target_veh_id_MT = carid[2365:2708]

# case_id_OF = caselist[2708:3276]
case_id_OF = [idx for idx in range(1, 569, 1)]
target_veh_id_OF = carid[2708:3276]

# case_id_EP0 = caselist[3276:3832]
case_id_EP0 = [idx for idx in range(1, 557, 1)]
target_veh_id_EP0 = carid[3276:3832]

# case_id_EP1 = caselist[3832:4129]
case_id_EP1 = [idx for idx in range(1, 298, 1)]
target_veh_id_EP1 = carid[3832:4129]

# case_id_GL= caselist[4129:7179]
case_id_GL = [idx for idx in range(1, 3051, 1)]
target_veh_id_GL = carid[4129:7179]

# case_id_MA= caselist[7179:8357]
case_id_MA = [idx for idx in range(1, 1179, 1)]
target_veh_id_MA = carid[7179:8357]

# case_id_EP= caselist[8357:8790]
case_id_EP = [idx for idx in range(1, 434, 1)]
target_veh_id_EP = carid[8357:8790]

# case_id_FT= caselist[8790:11242]
case_id_FT = [idx for idx in range(1, 2453, 1)]
target_veh_id_FT = carid[8790:11242]

# case_id_SR= caselist[11242:11794]
case_id_SR = [idx for idx in range(1, 553, 1)]
target_veh_id_SR = carid[11242:11794]
#---------------------------------------------------------------

MA_tv_info_dict = arrays_to_dict(case_id_MA, target_veh_id_MA)
ZS0_tv_info_dict = arrays_to_dict(case_id_ZS0, target_veh_id_ZS0)
ZS2_tv_info_dict = arrays_to_dict(case_id_ZS2, target_veh_id_ZS2)
LN_tv_info_dict = arrays_to_dict(case_id_LN, target_veh_id_LN)
MT_tv_info_dict = arrays_to_dict(case_id_MT, target_veh_id_MT)
OF_tv_info_dict = arrays_to_dict(case_id_OF, target_veh_id_OF)
EP0_tv_info_dict = arrays_to_dict(case_id_EP0, target_veh_id_EP0)
EP1_tv_info_dict = arrays_to_dict(case_id_EP1, target_veh_id_EP1)
GL_tv_info_dict = arrays_to_dict(case_id_GL, target_veh_id_GL)
EP_tv_info_dict = arrays_to_dict(case_id_EP, target_veh_id_EP)
FT_tv_info_dict = arrays_to_dict(case_id_FT, target_veh_id_FT)
SR_tv_info_dict = arrays_to_dict(case_id_SR, target_veh_id_SR)

with open('ZS2_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(ZS2_tv_info_dict, pickle_file) 

# # 保存为Pickle文件
with open('MA_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(MA_tv_info_dict, pickle_file)
with open('ZS0_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(ZS0_tv_info_dict, pickle_file)

with open('LN_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(LN_tv_info_dict, pickle_file)
with open('MT_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(MT_tv_info_dict, pickle_file)
with open('OF_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(OF_tv_info_dict, pickle_file)
with open('EP0_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(EP0_tv_info_dict, pickle_file)
with open('EP1_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(EP1_tv_info_dict, pickle_file)
with open('GL_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(GL_tv_info_dict, pickle_file)
with open('EP_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(EP_tv_info_dict, pickle_file)
with open('FT_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(FT_tv_info_dict, pickle_file)
with open('SR_tv_info_dict.pkl', 'wb') as pickle_file:
    pickle.dump(SR_tv_info_dict, pickle_file)

