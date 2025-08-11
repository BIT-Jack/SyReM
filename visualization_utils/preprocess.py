import pickle
list_value_task_labels = []
model = 'der_new'


for i in range(1,12):
    with open(f'./logging/{model}_bf_1000_STEP_SCORE_in_task_{i}.pkl', 'rb') as f:
        loaded_file = pickle.load(f)
    
    for j in range(1, len(loaded_file)+1):
        list_value_task_labels.append(loaded_file[j])




with open(f'./connected/{model}_bf_1000_STEP_SCORE_total.pkl', 'wb') as flist:
    pickle.dump(list_value_task_labels, flist)