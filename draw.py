import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import scipy.stats as stats
matplotlib.rcParams.update({'font.size': 16})

def cal_mean_std(data, min_len):
    data_new = []
    for line in data:
        data_new.append(line[:min_len])
    data_new = np.array(data_new)
    mean, std = data_new.mean(0), data_new.std(0)/np.sqrt(5)
    return mean, std

def fetch_file_list(log_name):
    file_pre = log_name
    # file_pre = 'slurm_logs/iso_403335_'    
    reward_all, reward_all_std = [], []
    for i in range(21, 65, 5):
        avg_reward = []
        min_len = 99
        for j in range(i,i+5):
            avg_reward_one = []
            file = file_pre + str(j) + '.log'
            with open(file, 'r') as f:
                f_h = f.readlines()
                for idx, line in enumerate(f_h):
                    l = line.strip().split()
                    if len(l)>2 and l[1]=='Generating':
                        r_line = f_h[idx-2].strip().split()[-1]
                        r_value = float(r_line)
                        avg_reward_one.append(r_value)
            min_len = min(min_len, len(avg_reward_one))
            avg_reward.append(avg_reward_one)
        mean, std = cal_mean_std(avg_reward, min_len)
            
        
        reward_all.append(mean)
        reward_all_std.append(std)
    # 0.01, 0.001, 0.1
    file_num = len(reward_all)
    sorted_file_list = []
    sorted_file_list_std = []

    for i in range(0,3):
        data_temp = []
        data_temp_std = []
        for j in range(i, file_num, 3):
            data_temp.append(reward_all[j])
            data_temp_std.append(reward_all_std[j])
        sorted_file_list.append(data_temp)
        sorted_file_list_std.append(data_temp_std)
    return sorted_file_list, sorted_file_list_std
    # for l in reward_all:
    #     print(l)


def draw_fig(data, data_std, name):
    fig = plt.figure(figsize=(10,5))
    mk = ('4', '+', '.', '2', '|', 4, '1', 5, 6, 7)
    colors = ('#e58e26', '#b71540', '#0c2461', '#0a3d62', '#079992', '#fad390', '#6a89cc','#60a3bc', '#78e08f')
        
    X = [0, 1, 2, 3]
    num = 4
    starting_point = min(data[0][0],data[1][0],data[2][0])
    diff0, diff1, diff2 = data[0][0]-starting_point,data[1][0]-starting_point,data[2][0]-starting_point
    diff0, diff1, diff2 =0,0,0
    plt.plot(X, data[0][:num]-diff0, label=r'$\lambda=0.001$', marker=mk[0],\
            color=colors[0],  linewidth=1)
    plt.fill_between(X, data[0][:num]-diff0 - data_std[0][:num], data[0][:num]-diff0 + data_std[0][:num], \
                    alpha=0.2, edgecolor=colors[0], facecolor=colors[0]) 
    plt.plot(X, data[1][:num]-diff1, label=r'$\lambda=0.01$', marker=mk[1],\
            color=colors[1],  linewidth=1)
    plt.fill_between(X, data[1][:num]-diff1 - data_std[1][:num], data[1][:num]-diff1 + data_std[1][:num], \
                    alpha=0.2, edgecolor=colors[1], facecolor=colors[1]) 
    plt.plot(X, data[2][:num]-diff2, label=r'$\lambda=0.1$', marker=mk[2],\
            color=colors[2],  linewidth=1)
    plt.fill_between(X, data[2][:num]-diff2 - data_std[2][:num], data[2][:num]-diff2 + data_std[2][:num], \
                    alpha=0.2, edgecolor=colors[2], facecolor=colors[2]) 

    min_, max_ = np.floor(min(min(data[0][:num]), min(data[1][:num]), min(data[2][:num]))), np.ceil(max(max(data[0][:num]), max(data[1][:num]), max(data[2][:num])))
    # plt.axis([0, 3, 0.0, 18])
    # plt.yticks(np.arange(0.0,18,2.0))   
    # min_ = 8
    plt.axis([0, 3, min_, max_])
    plt.yticks(np.arange(min_, max_,2.0))
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Average Return", fontsize=18)
    plt.grid(True,  linestyle='-.', linewidth=0.7)

    leg = plt.legend(loc=0, fancybox=True, fontsize=16)
    # leg.get_frame().set_alpha(0.5)
    # fig.savefig("datasize.png", bbox_inches='tight')
    fig.savefig("{}.pdf".format(name), bbox_inches='tight')



if __name__ == '__main__':
    log_name = 'slurm_logs/iso_403335_'
    # log_name = 'slurm_logs/iso_406919_'
    file, file_std = fetch_file_list(log_name)
    # draw_fig(file[0], file_std[0], 'oracle_pi_oracle_reward_rd')
    # draw_fig(file[1], file_std[1], 'oracle_pi_airl_reward_rd')
    # draw_fig(file[2], file_std[2], 'airl_pi_airl_reward_rd')

    draw_fig(file[0], file_std[0], 'oracle_pi_oracle_reward')
    draw_fig(file[1], file_std[1], 'oracle_pi_airl_reward')
    draw_fig(file[2], file_std[2], 'airl_pi_airl_reward')



