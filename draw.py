import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import scipy.stats as stats
matplotlib.rcParams.update({'font.size': 16})


def fetch_file_list():

    file_pre = 'slurm_logs/iso_401340_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'

    reward_all = []
    for i in range(10, 19, 1):
        avg_reward = []
        for j in range(i,i+1):
            file = file_pre + str(j) + '.log'
            with open(file, 'r') as f:
                f_h = f.readlines()
                for idx, line in enumerate(f_h):
                    l = line.strip().split()
                    if len(l)>2 and l[1]=='Generating':
                        r_line = f_h[idx-2].strip().split()[-1]
                        r_value = float(r_line)
                        avg_reward.append(r_value)
        reward_all.append(avg_reward)
    # 0.01, 0.001, 0.1
    file_num = len(reward_all)
    sorted_file_list = []
    for i in range(0,3):
        data_temp = []
        for j in range(i, file_num, 3):
            data_temp.append(reward_all[j])
        sorted_file_list.append(data_temp)
    return sorted_file_list
    # for l in reward_all:
    #     print(l)


def draw_fig(data, name):
    fig = plt.figure(figsize=(10,5))
    mk = ('4', '+', '.', '2', '|', 4, '1', 5, 6, 7)
    colors = ('#e58e26', '#b71540', '#0c2461', '#0a3d62', '#079992', '#fad390', '#6a89cc','#60a3bc', '#78e08f')
        
    X = [0, 1, 2, 3]


    plt.plot(X, data[1][:4], label='lambda=0.001', marker=mk[0],\
            color=colors[0],  linewidth=3)
    plt.plot(X, data[0][:4], label='lambda=0.01', marker=mk[1],\
            color=colors[1],  linewidth=3)
    plt.plot(X, data[2][:4], label='lambda=0.1', marker=mk[2],\
            color=colors[2],  linewidth=3)

 
    plt.axis([0, 3, 0.0, 18])
    plt.yticks(np.arange(0.0,18,2.0))
    # plt.xticks(np.arange(0,self.length*1000,self.length*1000//250 * 10))
    plt.xticks([0, 1, 2, 3])
    # plt.xticks([1000] + np.arange(10000,self.length*1000, 5000).tolist())
    plt.xlabel("Optimizing Turn", fontsize=18)
    plt.ylabel("Avg Traj Turn", fontsize=18)
    plt.grid(True,  linestyle='-.', linewidth=0.7)

    leg = plt.legend(loc=0, fancybox=True, fontsize=16)
    # leg.get_frame().set_alpha(0.5)
    # fig.savefig("datasize.png", bbox_inches='tight')
    fig.savefig("{}.pdf".format(name), bbox_inches='tight')



if __name__ == '__main__':
    file = fetch_file_list()
    draw_fig(file[0], 'oracle_pi_oracle_reward')
    draw_fig(file[1], 'oracle_pi_airl_reward')
    draw_fig(file[2], 'airl_pi_airl_reward')





