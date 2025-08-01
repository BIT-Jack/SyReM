import numpy as np
import matplotlib.pyplot as plt
from utils.args_loading import *
from utils.metrics import *
import pickle
limflag = True
ylim_fde = (-0.5, 3.5)
ylim_mr = (-5, 50)
ylim_im = (-1, 1)
ylim_im_mr = (-20, 10)

def results_loading(result_direction,  model_name, metrics_to_cal, buffer_size, total_task_num):
    if metrics_to_cal == 'bwt':
        bwt = {}
        

        for lt in range(2, total_task_num+1):

            bwt[lt] = {}
            
            fde_bwt, mr_bwt = calculate_backward_transfer(result_direction, lt,  model_name, buffer_size)
            bwt[lt]['fde_bwt'] = fde_bwt
            bwt[lt]['mr_bwt'] = mr_bwt
        
        return bwt
    elif metrics_to_cal == 'fwt':
        fwt_jt = {}
        fwt_rd = {}

        for lt in range(1, total_task_num):
            fwt_jt[lt] = {}

            fde_fwt_jt, mr_fwt_jt = calculate_forward_transfer_ref_joint(result_direction, lt, total_task_num, model_name, buffer_size)
            fwt_jt[lt]['fde_fwt'] = fde_fwt_jt
            fwt_jt[lt]['mr_fwt'] = mr_fwt_jt
        
        for lt in range(1, total_task_num):
            fwt_rd[lt] = {}

            fde_fwt_rd, mr_fwt_rd = calculate_forward_transfer_ref_random(result_direction, lt, total_task_num, model_name, buffer_size)
            fwt_rd[lt]['fde_fwt'] = fde_fwt_rd
            fwt_rd[lt]['mr_fwt'] = mr_fwt_rd

        return fwt_jt, fwt_rd
    
    elif metrics_to_cal == 'im':
        im = {}

        for lt in range(1, total_task_num+1):
            im[lt] = {}
            fde_im, mr_im = calculate_intransigence_measure_for_task_k(result_direction, lt, model_name, buffer_size)
            im[lt]['fde_im'] = fde_im
            im[lt]['mr_im'] = mr_im


        return im
    
    else:
        jt_test = {}
        
        for lt in range(1, total_task_num+1):
            jt_test[lt] = {}
            with open(result_direction+ f'/{model_name}_bf_{buffer_size}_learn_{lt}_jt_12.pkl', 'rb') as f:
                loaded_result = pickle.load(f)
            fde_jt = loaded_result['fde']
            mr_jt = loaded_result['mr']

            jt_test[lt]['fde_jt'] = fde_jt
            jt_test[lt]['mr_jt'] = mr_jt
        return jt_test



    


def results_plotting(metrics_to_cal, results_all, model_list,  size_of_figure =(7.16, 4.4)):
    if metrics_to_cal == 'bwt':
    
        plt.figure(figsize=size_of_figure)
        
        for name in model_list:
            x = [i for i in range(2, len(results_all[name])+2)] 
            y = []
            for i in range(2, len(results_all[name])+2):
                y.append(results_all[name][i]['fde_bwt'])

            
            
            plt.plot(x, y, marker='o', label=name)
        plt.xlabel('number of learnt tasks')
        plt.ylabel('FDE Backward Transfer')
        # plt.title('FDE Backward Transfer by Method')
        if limflag:
            plt.ylim(ylim_fde)
        plt.legend()
        plt.grid(True)
        plt.savefig(logging_dir+'/FDE-BWT.png')

        plt.figure(figsize=size_of_figure)
        
        for name in model_list:
            x = [i for i in range(2, len(results_all[name])+2)] 
            y = []
            for i in range(2, len(results_all[name])+2):
                y.append(results_all[name][i]['mr_bwt'])

            
            
            plt.plot(x, y, marker='o', label=name)

        plt.xlabel('number of learnt tasks')
        plt.ylabel('Miss Rate Backward Transfer')
        # plt.title('MR Backward Transfer by Method')
        if limflag:
            plt.ylim(ylim_mr)
        plt.legend()
        plt.grid(True)
        plt.savefig(logging_dir+'/MR-BWT.png')


    elif metrics_to_cal == 'fwt':
        plt.figure(figsize=size_of_figure)
        
        for name in model_list:
            x = [i for i in range(1, len(results_all[name])+1)] 
            y = []
            for i in range(1, len(results_all[name])+1):
                y.append(results_all[name][i]['fde_fwt'])

            
            
            plt.plot(x, y, marker='o', label=name)
        plt.xlabel('number of learnt tasks')
        plt.ylabel('FDE Forward Transfer')
        # plt.title('FDE Forward Transfer Reference')
        if limflag:
            plt.ylim(ylim_fde)
        plt.legend()
        plt.grid(True)
        plt.savefig(logging_dir+'/FDE-FWT.png')

        plt.figure(figsize=size_of_figure)
        
        for name in model_list:
            x = [i for i in range(1, len(results_all[name])+1)] 
            y = []
            for i in range(1, len(results_all[name])+1):
                y.append(results_all[name][i]['mr_fwt'])

            
            
            plt.plot(x, y, marker='o', label=name)
        plt.xlabel('number of learnt tasks')
        plt.ylabel('Miss Rate Forward Transfer')
        # plt.title('FDE Forward Transfer Reference')
        if limflag:
            plt.ylim(ylim_mr)
        plt.legend()
        plt.grid(True)
        plt.savefig(logging_dir+'/MR-FWT.png')

    elif metrics_to_cal == 'im':

        bar_width = 0.2
        x = np.arange(1, len(results_all[model_list[0]])+1)

        fig, ax = plt.subplots(figsize=(12, 6))

        # 遍历每个模型，绘制柱子
        for i, model in enumerate(model_list):
            # 获取当前模型的 fde_im 数据
            values = [results_all[model].get(task_id, {}).get('fde_im', 0) for task_id in x]
            # 绘制每组的柱子
            ax.bar(x + i * bar_width, values, bar_width, label=model)

        # 添加图形标签和标题
        ax.set_xlabel('Task ID')
        ax.set_ylabel('FDE IM Value')
        ax.set_title('Bar Chart of FDE_IM for Models')
        ax.set_xticks(x + bar_width * (len(model_list) - 1) / 2)
        ax.set_xticklabels(x)
        ax.legend(title="Models")
        if limflag:

            plt.ylim(ylim_im)   
        plt.savefig(logging_dir+'/FDE-IM.png')
    

        fig, ax = plt.subplots(figsize=(12, 6))

        # 遍历每个模型，绘制柱子
        for i, model in enumerate(model_list):
            # 获取当前模型的 fde_im 数据
            values = [results_all[model].get(task_id, {}).get('mr_im', 0) for task_id in x]
            # 绘制每组的柱子
            ax.bar(x + i * bar_width, values, bar_width, label=model)

        # 添加图形标签和标题
        ax.set_xlabel('Task ID')
        ax.set_ylabel('MR IM Value')
        ax.set_title('Bar Chart of MissRate_IM for Models')
        ax.set_xticks(x + bar_width * (len(model_list) - 1) / 2)
        ax.set_xticklabels(x)
        ax.legend(title="Models")
        if limflag:
        
            plt.ylim(ylim_im_mr)
        plt.savefig(logging_dir+'/MR-IM.png')



    else: #joint testing
        plt.figure(figsize=size_of_figure)
                
        for name in model_list:
            x = [i for i in range(1, len(results_all[name])+1)] 
            y = []
            for i in range(1, len(results_all[name])+1):
                y.append(results_all[name][i]['fde_jt'])

            
            
            plt.plot(x, y, marker='o', label=name)
        plt.xlabel('number of learnt tasks')
        plt.ylabel('FDE in the joint test set')
        # plt.title('FDE Forward Transfer Reference')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=size_of_figure)
                
        for name in model_list:
            x = [i for i in range(1, len(results_all[name])+1)] 
            y = []
            for i in range(1, len(results_all[name])+1):
                y.append(results_all[name][i]['mr_jt'])

            
            
            plt.plot(x, y, marker='o', label=name)
        plt.xlabel('number of learnt tasks')
        plt.ylabel('Miss Rate in the joint test set')
        # plt.title('FDE Forward Transfer Reference')
        plt.legend()
        plt.grid(True)


    

        

    plt.show()














bf = 1000


name1 = 'gss'
name2 = 'gssaux'
name3 = 'gss'

num_tasks = 12

bwt_model1 = results_loading(result_dir, name1, 'bwt', bf, num_tasks)
bwt_model2 = results_loading(result_dir, name2, 'bwt', bf, num_tasks)
bwt_model3 = results_loading(result_dir, name3, 'bwt', bf, num_tasks)
bwt_all = {name1:bwt_model1, name2:bwt_model2, name3:bwt_model3}

fwt_jt_model1, fwt_rd_vanilla = results_loading(result_dir, name1, 'fwt', bf, num_tasks) 
fwt_jt_model2, fwt_rd_der = results_loading(result_dir, name2, 'fwt', bf, num_tasks)
fwt_jt_model3, fwt_rd_derccl = results_loading(result_dir, name3, 'fwt', bf, num_tasks)
fwt_all = {name1:fwt_jt_model1, name2:fwt_jt_model2, name3:fwt_jt_model3}

im_model1 = results_loading(result_dir, name1, 'im', bf, num_tasks)
im_model2 = results_loading(result_dir, name2, 'im', bf, num_tasks)
im_model3 = results_loading(result_dir, name3, 'im', bf, num_tasks)
im_all = {name1: im_model1, name2:im_model2, name3:im_model3}





plot_list=[name1, name2]

results_plotting('bwt', bwt_all, plot_list)

results_plotting('fwt', fwt_all, plot_list)

results_plotting('im', im_all, plot_list)

# results_plotting('jt_test', jt_all, plot_list)






