import pickle
import pandas as pd
import numpy as np
import os
#READ ME
#This python file is used to extract original information of target vehicles in validation sets (you can also directly use the processed data saved in ./logging/original_reference)
#The main goal is to obtain the original coordinates of target vehicles from .csv files, using for coordinate transfer in map visulization
#inputs 1: please download the raw data from INTERACTION official website (.csv files)
#inputs 2: load the "XX_tv_info_dict.pkl" (replace XX as the specific scenario name) from file folder ./logging/original_reference
# "XX_tv_info_dict.pkl" was obtianed by running ./visualization_utils/dictionary.py     which builds a dictionary from case id to target vehicle id

def extract_specified_car_data(df, track_id_dict):
    # filter 'car' type cases
    cars = df[df['agent_type'] == 'car']
    
    # obtain the 10th timeframe (the current time in each case)
    tenth_frames = cars[cars['frame_id'] == 10]
    
    result = pd.DataFrame(columns=['case_id', 'track_id', 'x', 'y', 'psi_rad', 'length', 'width'])
    
    for case_id, track_id in track_id_dict.items():
        data = tenth_frames[(tenth_frames['case_id'] == int(case_id)) & (tenth_frames['track_id'] == int(track_id))]
        
        if not data.empty:
            result = pd.concat([result, data[['case_id', 'track_id', 'x', 'y', 'psi_rad', 'length', 'width']]], ignore_index=True)
    
    return result

def process_files(input_directory, track_id_dict):
    # for filename in os.listdir(input_directory):
    #     if filename.endswith('.csv'):
    #         file_path = os.path.join(input_directory, filename)
    file_path = input_directory
    df = pd.read_csv(file_path)
    
    data = extract_specified_car_data(df, track_id_dict)
    
    data = data.sort_values(by=['case_id'])
    
    npz_filename = scenario + '.npz'
    np.savez_compressed(npz_filename, case_id=data['case_id'].values,
                        track_id=data['track_id'].values,
                        x=data['x'].values,
                        y=data['y'].values,
                        psi_rad=data['psi_rad'].values,
                        length=data['length'].values,
                        width=data['width'].values)
    print(f"Processed and saved {npz_filename}")

#main (example)
input_directory = '.raw_datasets/val/DR_USA_Roundabout_SR_val.csv'  # directory of raw csv files
scenario = 'SR_target'
with open('SR_tv_info_dict.pkl', 'rb') as pickle_file:
    track_id_dict = pickle.load(pickle_file)


process_files(input_directory, track_id_dict)



print("yes")