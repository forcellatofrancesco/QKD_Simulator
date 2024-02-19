import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import csv
import json
import seaborn as sb
import math
import glob
from statistics import median, mean
import networkx as nx


def deliv(df, dist, rate_list):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_del_load_1 = df[(df['Sim. Command'].str.contains('--mess-rate '+rate_list[0]+'$')) & (df['Delivered'] == True)]
    df_del_load_2 = df[(df['Sim. Command'].str.contains('--mess-rate '+rate_list[1]+'$')) & (df['Delivered'] == True)]
    df_del_load_3 = df[(df['Sim. Command'].str.contains('--mess-rate '+rate_list[2]+'$')) & (df['Delivered'] == True)]
    df_del_load_4 = df[(df['Sim. Command'].str.contains('--mess-rate '+rate_list[3]+'$')) & (df['Delivered'] == True)]
    df_del_load_5 = df[(df['Sim. Command'].str.contains('--mess-rate '+rate_list[4]+'$')) & (df['Delivered'] == True)]

    return [len(df_del_load_1),len(df_del_load_2),len(df_del_load_3),len(df_del_load_4),len(df_del_load_5)]

def ber(df, dist):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_ber = df[(df['qchannel'] == 'qchannel0_0to1')]

    return df_ber['ber'].mean()


def tot_keys(df, dist):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_1 = df['tr_node0_to_node1']

    return df_1.tolist()

def disc_bit(df, dist):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_1 = df['tr_node0_to_node1']

    return df_1.tolist()

def disc_bit_rate(df, dist, rate_list):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_del_load_1 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[0]+'$')]
    df_del_load_2 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[1]+'$')]
    df_del_load_3 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[2]+'$')]
    df_del_load_4 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[3]+'$')]
    df_del_load_5 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[4]+'$')]

    return [df_del_load_1['tr_node1_to_node0'],df_del_load_2['tr_node1_to_node0'],df_del_load_3['tr_node1_to_node0'],df_del_load_4['tr_node1_to_node0'],df_del_load_5['tr_node1_to_node0']]

def ric_bit_rate(df, dist, rate_list):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_del_load_1 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[0]+'$')]
    df_del_load_2 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[1]+'$')]
    df_del_load_3 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[2]+'$')]
    df_del_load_4 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[3]+'$')]
    df_del_load_5 = df[df['Sim. Command'].str.contains('--mess-rate '+rate_list[4]+'$')]

    return [df_del_load_1['tr_node1_to_node0'],df_del_load_2['tr_node1_to_node0'],df_del_load_3['tr_node1_to_node0'],df_del_load_4['tr_node1_to_node0'],df_del_load_5['tr_node1_to_node0']]


def ric_bit(df, dist):
    df = df[df['Sim. Command'].str.contains(dist)]

    df_1 = df['tr_node1_to_node0']

    return df_1.tolist()


def plot4(dataset):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    
    joined_files = []
    for d in dataset:
        joined_files.append(sim_dir + d + '/packet_result.csv')
    df1 = pd.concat(map(pd.read_csv, joined_files), ignore_index=True)

    dim_k = 128

    rate_list_p = [1000 * dim_k, 2000 * dim_k, 2500 * dim_k, 3000 * dim_k, 3500 * dim_k]
    rate_list = ['0.001', '0.0005', '0.0004', '0.00033333', '0.0002857142857']

    res_2_d = [x * dim_k for x in deliv(df1, '-3000m.', rate_list)]
    res_3_d = [x * dim_k for x in deliv(df1, '-4000m.', rate_list)]
    res_4_d = [x * dim_k for x in deliv(df1, '-5000m.', rate_list)]

    #ber_s = [ber(df2, 'chain_2_node_seq_4000.'), ber(df2, 'chain_3_node_seq_4000.'), ber(df2, 'chain_4_node_seq_4000.')]

    # fig = plt.figure(200)
    # fig.subplots_adjust(hspace=0.4, wspace=0.1)
    # fig.set_figwidth(12)
    # fig.set_figheight(6)

    # plt.subplot(1, 1, 1)
    # plt.plot([2,3,4], [res_2_d,res_3_d,res_4_d], marker='o')
    # i = 0
    # for x, y in zip([2,3,4], [res_2_d,res_3_d,res_4_d]):
    #     plt.annotate(text='BER ' + str(ber_s[i]), xy=(x, y))
    #     i+=1
    # plt.yticks([res_2_d,res_3_d,res_4_d])
    # plt.grid()
    # plt.xlabel("Meters")
    # plt.ylabel("Key rate (bit/s)")
    # plt.title('Q.C. 1Mbps - key 128 bit')

    fig = plt.figure(300)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.set_figwidth(12)
    fig.set_figheight(10)

    plt.subplot(3, 1, 1)
    plt.plot(rate_list_p, res_2_d, marker='o')
    for x, y in zip(rate_list_p, res_2_d):
        plt.annotate(text=str(y)+' bit', xy=(x, y))
    plt.grid()
    plt.xlabel("Bit-rate (bit/s)")
    plt.ylabel("Delivered bits")
    plt.title('C.C. 20Mbps (Q.C 3000m)')

    plt.subplot(3, 1, 2)
    plt.plot(rate_list_p, res_3_d, marker='o')
    for x, y in zip(rate_list_p, res_3_d):
        plt.annotate(text=str(y)+' bit', xy=(x, y))
    plt.grid()
    plt.xlabel("Bit-rate (bit/s)")
    plt.ylabel("Delivered bits")
    plt.title('C.C. 20Mbps (Q.C 4000m)')

    plt.subplot(3, 1, 3)
    plt.plot(rate_list_p, res_4_d, marker='o')
    for x, y in zip(rate_list_p, res_4_d):
        plt.annotate(text=str(y)+' bit', xy=(x, y))
    plt.grid()
    plt.xlabel("Bit-rate (bit/s)")
    plt.ylabel("Delivered bits")
    plt.title('C.C. 20Mbps (Q.C 5000m)')

    plt.show()



def plot3(dataset):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    joined_files = []

    for d in dataset:
        joined_files.append(sim_dir + d + '/tot_keys.csv')
    df = pd.concat(map(pd.read_csv, joined_files), ignore_index=True)

    joined_files = []
    for d in dataset:
        joined_files.append(sim_dir + d + '/ber_dist.csv')
    df3 = pd.concat(map(pd.read_csv, joined_files), ignore_index=True)

    dim_k = 128

    ber_s = [ber(df3, '-2000m.'), ber(df3, '-3000m.'), ber(df3, '-4000m.'), ber(df3, '-5000m.')]

    res_2000 = [x*dim_k for x in tot_keys(df, '-2000m.')]
    res_3000 = [x*dim_k for x in tot_keys(df, '-3000m.')]
    res_4000 = [x*dim_k for x in tot_keys(df, '-4000m.')]
    res_5000 = [x*dim_k for x in tot_keys(df, '-5000m.')]

    print(res_3000)
    print(res_4000)
    print(res_5000)

    fig = plt.figure(200)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    plt.boxplot([res_2000, res_3000, res_4000, res_5000])
    plt.xticks([1,2,3,4], ["2000","3000","4000","5000"])
    i = 0
    for x, y in zip([1,2,3,4], [res_2000[0],res_3000[0],res_4000[0],res_5000[0]]):
        plt.annotate(text='BER ' + str(ber_s[i]), xy=(x, y))
        i+=1
    plt.grid()
    plt.xlabel("Meters")
    plt.ylabel("Key rate (bit/s)")
    plt.title('Q.C. 1Mbps - key 128 bit')

    plt.show()

def plot_cc(d):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    df = pd.read_csv(sim_dir + d + '/tot_bit_cc.csv')

    col = df.columns.values.tolist()[1:]

    v = []
    for c in col:
        v.append(df[c].values[0]*128)
    
    fig = plt.figure(300)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.set_figwidth(12)
    fig.set_figheight(10)

    plt.bar([i for i in range(len(v))],v)
    
    plt.grid()
    plt.xlabel("Link")
    plt.ylabel("Transmitted bits")
    plt.title('Transmitted bit links')

    plt.plot([i for i in range(len(v))], v, color='r')


    plt.show()

def plot_cc_box_plot(dataset):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    joined_files = []

    for d in dataset:
        joined_files.append(sim_dir + d + '/tot_bit_cc.csv')
    df = pd.concat(map(pd.read_csv, joined_files), ignore_index=True)

    df_heavy = df[df['Sim. Command'].str.contains('--mess-rate 0.002')]
    df_low = df[df['Sim. Command'].str.contains('--mess-rate 0.04')]

    col = df_heavy.columns.values.tolist()[1:]

    v_heavy = []
    v_low = []
    for c in col:
        v_heavy.append(df_heavy[c]*128)
        v_low.append(df_low[c]*128)

    fig = plt.figure(300)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.set_figwidth(12)
    fig.set_figheight(10)

    plt.subplot(2, 1, 1)
    plt.boxplot(v_heavy)
    plt.bar([i for i in range(1,len(v_heavy)+1)], [median(l) for l in v_heavy],alpha=0.3)
    plt.grid()
    plt.title("Heavy Load")
    plt.xlabel("Links")
    plt.ylabel("Transmitted bits")

    plt.subplot(2, 1, 2)
    plt.boxplot(v_low)
    plt.bar([i for i in range(1,len(v_low)+1)], [median(l) for l in v_low],alpha=0.3)
    plt.grid()
    plt.title("Low Load")
    plt.xlabel("Links")
    plt.ylabel("Transmitted bits")
    
    plt.show()

def prova(ds):

    sim_dir = os.path.dirname('project/simulations/') + '/'
    c = ['node0','node1','node2','node3','node4','node5','node6','node7','node8','node9']

    v_heavy = [[],[],[],[],[],[],[],[],[],[]]
    v_low = [[],[],[],[],[],[],[],[],[],[]]

    for d in ds:
        df = pd.read_csv(sim_dir + d + '/packet_result.csv')
        v = []
        for i in c:
            v.append(len(df[(df['Delivered'] == True) & (df['Destination'] == i)]))

        #heavy
        if '--mess-rate 0.002' in df['Sim. Command'][0]:
            for i in range(10):
                v_heavy[i].append(v[i]*128)
        else:
            for i in range(10):
                v_low[i].append(v[i]*128)
    
    fig = plt.figure(300)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.set_figwidth(12)
    fig.set_figheight(10)

    plt.subplot(2, 1, 1)
    plt.boxplot(v_heavy)
    plt.bar([i for i in range(1,len(v_heavy)+1)], [median(l) for l in v_heavy],alpha=0.3)
    plt.grid()
    plt.title("Heavy Load")
    plt.xlabel("Nodes")
    plt.ylabel("Received bits")

    plt.subplot(2, 1, 2)
    plt.boxplot(v_low)
    plt.bar([i for i in range(1,len(v_low)+1)], [median(l) for l in v_low],alpha=0.3)
    plt.grid()
    plt.title("Low Load")
    plt.xlabel("Nodes")
    plt.ylabel("Received bits")
    
    plt.show()

def heat_map(ds):

    sim_dir = os.path.dirname('project/simulations/') + '/'
    c = ['node0','node1','node2','node3','node4','node5','node6','node7','node8','node9']

    v = np.zeros((10,10),dtype=object)
    v2 = np.zeros((10,10),dtype=object)
    for i in range(10):
        for j in range(10):
            v[i][j] = []
            v2[i][j] = []

    for d in ds:
        df = pd.read_csv(sim_dir + d + '/packet_result.csv')
        row = col = 0
        if '--mess-rate 0.002' in df['Sim. Command'][0]:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1
        else:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v2[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1


    for i in range(10):
        for j in range(10):
            if v[i][j] != []:
                v[i][j] = median(v[i][j])
            else:
                v[i][j] = 0
    
    for i in range(10):
        for j in range(10):
            if v2[i][j] != []:
                v2[i][j] = median(v2[i][j])
            else:
                v2[i][j] = 0

    v = np.rot90(v, 3)
    v2 = np.rot90(v2, 3)

    a = np.zeros((10,10))
    b = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            a[i][j] = v[i][j]
            b[i][j] = v2[i][j]
    
    fig = plt.figure(300)
    #fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.set_figwidth(15)
    fig.set_figheight(6)

    plt.subplot(1, 2, 1)
    sb.set()
    ax = sb.heatmap(a, yticklabels=[9,8,7,6,5,4,3,2,1,0], cmap='Blues', annot=True, annot_kws={"size": 7})
    ax.set(xlabel='Source Nodes', ylabel='Destination Nodes')
    plt.title("Heavy Load - Delivered Bits")

    plt.subplot(1, 2, 2)
    sb.set()
    ax = sb.heatmap(b, yticklabels=[9,8,7,6,5,4,3,2,1,0], cmap='Blues', annot=True, annot_kws={"size": 7})
    ax.set(xlabel='Source Nodes', ylabel='Destination Nodes')
    plt.title("Low Load - Delivered Bits")

    plt.show()
    

def plot_prob(ds):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    c = ['node0','node1','node2','node3','node4','node5','node6','node7','node8','node9']

    v = np.zeros((10,10),dtype=object)
    v2 = np.zeros((10,10),dtype=object)
    for i in range(10):
        for j in range(10):
            v[i][j] = []
            v2[i][j] = []

    for d in ds:
        df = pd.read_csv(sim_dir + d + '/packet_result.csv')
        row = col = 0
        if '--mess-rate 0.002' in df['Sim. Command'][0]:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1
        else:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v2[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1


    for i in range(10):
        for j in range(10):
            if v[i][j] != []:
                v[i][j] = median(v[i][j])
            else:
                v[i][j] = 0
    
    for i in range(10):
        for j in range(10):
            if v2[i][j] != []:
                v2[i][j] = median(v2[i][j])
            else:
                v2[i][j] = 0

    v = np.rot90(v, 3)
    v2 = np.rot90(v2, 3)

    pro = []
    num_camm = []
    for p in range(1,10):
        aux = []
        c = 0
        for i in range(9,-1,-1):
            if i + p <= 9:
                aux.append(v[i+p][c])
                #print(v[i+p][c])
            if i - p >= 0:
                aux.append(v[i-p][c])
                #print(v[i-p][c])
            c += 1
        pro.append(mean(aux))
        num_camm.append(len(aux))
    
    pro1 = []
    num_camm1 = []
    for p in range(1,10):
        aux = []
        c = 0
        for i in range(9,-1,-1):
            if i + p <= 9:
                aux.append(v2[i+p][c])
                print(i+p, c)
            if i - p >= 0:
                aux.append(v2[i-p][c])
                print(i-p,c)
            c += 1
        pro1.append(mean(aux))
        num_camm1.append(len(aux))

    fig = plt.figure(300)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.set_figwidth(10)
    fig.set_figheight(10)

    ax1 = plt.subplot(2, 1, 1)
    plt.bar([i for i in range(1,len(pro)+1)], pro,alpha=0.3)
    plt.grid()
    plt.xticks([1,2,3,4,5,6,7,8,9])
    plt.xlabel("Path length")
    plt.ylabel("Average Received bits")
    ax2 = ax1.twinx()
    ax2.plot([1,2,3,4,5,6,7,8,9], num_camm, '.-', color='red')
    plt.title("Heavy Load")
    plt.ylabel("Number of Paths", color='red')
    

    ax1 = plt.subplot(2, 1, 2)
    plt.bar([i for i in range(1,len(pro1)+1)], pro1,alpha=0.3)
    plt.grid()
    plt.xticks([1,2,3,4,5,6,7,8,9])
    plt.title("Low Load")
    plt.xlabel("Path length")
    plt.ylabel("Average Received bits")
    ax2 = ax1.twinx()
    ax2.plot([1,2,3,4,5,6,7,8,9], num_camm1, '.-', color='red')
    plt.ylabel("Number of Paths", color='red')
    


    plt.show()


def plot_prob_new(ds):
    sim_dir = os.path.dirname('project/simulations/') + '/'
    c = ['node0','node1','node2','node3','node4','node5','node6','node7','node8','node9']

    v = np.zeros((10,10),dtype=object)
    v2 = np.zeros((10,10),dtype=object)
    v3 = np.zeros((10,10),dtype=object)
    v4 = np.zeros((10,10),dtype=object)
    for i in range(10):
        for j in range(10):
            v[i][j] = []
            v2[i][j] = []
            v3[i][j] = []
            v4[i][j] = []
    
    for d in ds:
        df = pd.read_csv(sim_dir + d + '/packet_result.csv')
        row = col = 0
        if '--mess-rate 0.002' in df['Sim. Command'][0]:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1
        else:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v2[row][col].append(len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1

    for d in ds:
        df = pd.read_csv(sim_dir + d + '/packet_result.csv')
        row = col = 0
        if '--mess-rate 0.002' in df['Sim. Command'][0]:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v3[row][col].append(len(df[(df['Sent'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1
        else:
            for i in c:
                col = 0
                for j in c:
                    if i != j:
                        v4[row][col].append(len(df[(df['Sent'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128)
                    col += 1
                row += 1


    for i in range(10):
        for j in range(10):
            if v[i][j] != []:
                v[i][j] = median(v[i][j])
            else:
                v[i][j] = 0
    
    for i in range(10):
        for j in range(10):
            if v2[i][j] != []:
                v2[i][j] = median(v2[i][j])
            else:
                v2[i][j] = 0
    
    for i in range(10):
        for j in range(10):
            if v3[i][j] != []:
                v3[i][j] = median(v3[i][j])
            else:
                v3[i][j] = 0
    
    for i in range(10):
        for j in range(10):
            if v4[i][j] != []:
                v4[i][j] = median(v4[i][j])
            else:
                v4[i][j] = 0

    v = np.rot90(v, 3)
    v2 = np.rot90(v2, 3)
    v3 = np.rot90(v3, 3)
    v4 = np.rot90(v4, 3)

    v_l = np.zeros((10,10))
    v_h = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            if v[i][j] != 0:
                v_h[i][j] = v[i][j]/v3[i][j]
    
    for i in range(10):
        for j in range(10):
            if v2[i][j] != 0:
                v_l[i][j] = v2[i][j]/v4[i][j]
    
    fig = plt.figure(300)
    #fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.set_figwidth(15)
    fig.set_figheight(6)

    plt.subplot(1, 2, 1)
    sb.set()
    ax = sb.heatmap(v_h, yticklabels=[9,8,7,6,5,4,3,2,1,0], cmap='Blues', annot=True, annot_kws={"size": 7})
    ax.set(xlabel='Source Nodes', ylabel='Destination Nodes')
    plt.title("Heavy Load - Delivered Bits")

    plt.subplot(1, 2, 2)
    sb.set()
    ax = sb.heatmap(v_l, yticklabels=[9,8,7,6,5,4,3,2,1,0], cmap='Blues', annot=True, annot_kws={"size": 7})
    ax.set(xlabel='Source Nodes', ylabel='Destination Nodes')
    plt.title("Low Load - Delivered Bits")

    plt.show()

    # pro = []
    # for p in range(1,10):
    #     aux = []
    #     c = 0
    #     for i in range(9,-1,-1):
    #         if i + p <= 9:
    #             aux.append(v[i+p][c])
    #             #print(v[i+p][c])
    #         if i - p >= 0:
    #             aux.append(v[i-p][c])
    #             #print(v[i-p][c])
    #         c += 1
    #     pro.append(mean(aux))
    
    # pro1 = []
    # for p in range(1,10):
    #     aux = []
    #     c = 0
    #     for i in range(9,-1,-1):
    #         if i + p <= 9:
    #             aux.append(v2[i+p][c])
    #             print(i+p, c)
    #         if i - p >= 0:
    #             aux.append(v2[i-p][c])
    #             print(i-p,c)
    #         c += 1
    #     pro1.append(mean(aux))
    
    # pro2 = []
    # for p in range(1,10):
    #     aux = []
    #     c = 0
    #     for i in range(9,-1,-1):
    #         if i + p <= 9:
    #             aux.append(v3[i+p][c])
    #             print(i+p, c)
    #         if i - p >= 0:
    #             aux.append(v3[i-p][c])
    #             print(i-p,c)
    #         c += 1
    #     pro2.append(mean(aux))
    
    # pro3 = []
    # for p in range(1,10):
    #     aux = []
    #     c = 0
    #     for i in range(9,-1,-1):
    #         if i + p <= 9:
    #             aux.append(v4[i+p][c])
    #             print(i+p, c)
    #         if i - p >= 0:
    #             aux.append(v4[i-p][c])
    #             print(i-p,c)
    #         c += 1
    #     pro3.append(mean(aux))
    
    # res1 = [pro[i]/pro2[i] for i in range(len(pro))]
    # res2 = [pro1[i]/pro3[i] for i in range(len(pro1))]

    # fig = plt.figure(300)
    # fig.subplots_adjust(hspace=0.2, wspace=0.1)
    # fig.set_figwidth(10)
    # fig.set_figheight(10)

    # plt.subplot(2, 1, 1)
    # plt.bar([i for i in range(1,len(res1)+1)], res1,alpha=0.3)
    # plt.grid()
    # plt.xticks([1,2,3,4,5,6,7,8,9])
    # plt.title("Heavy Load")
    # plt.xlabel("Path length")
    # plt.ylabel("Probability of path success")
    
    # plt.subplot(2, 1, 2)
    # plt.bar([i for i in range(1,len(res2)+1)], res2,alpha=0.3)
    # plt.grid()
    # plt.xticks([1,2,3,4,5,6,7,8,9])
    # plt.title("Low Load")
    # plt.xlabel("Path length")
    # plt.ylabel("Probability of path success")

    # plt.show()
    

def len_path(p, l):
    return [i for i in p if len(i)-1 == l]

def num_path(p):
    res = {}
    for i in range(1,14):
        res[i] = 0

    for i in range(2,15):
        for j in p:
            if len(j) == i:
                res[i-1] += 1
    
    return {k: v for k, v in res.items() if v != 0}

def path(d):
    
    data = json.load(open(os.path.dirname('project/file/') + '/graph_networkx_15_nodes.json'))
    g = nx.node_link_graph(data)

    bet_c = nx.betweenness_centrality(g)

    print(bet_c)

    c = ['node0','node1','node2','node3','node4','node5','node6','node7','node8','node9','node10','node11','node12','node13','node14']

    v = np.zeros((15,15))
    v1 = np.zeros((15,15))
    
    p = []
    for i in range(15):
        for j in range(15):
            if i != j:
                for path in nx.all_simple_paths(g,i,j):
                    p.append(path)

    df = pd.read_csv(os.path.dirname('project/simulations/') + "/" + d + '/packet_result.csv')
    row = col = 0
    
    for i in c:
        col = 0
        for j in c:
            if i != j:
                v[row][col] = len(df[(df['Delivered'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128
            col += 1
        row += 1
    
    row = col = 0
    
    for i in c:
        col = 0
        for j in c:
            if i != j:
                v1[row][col] = len(df[(df['Sent'] == True) & (df['Source'] == i) & (df['Destination'] == j)])*128
            col += 1
        row += 1
    
    v = np.rot90(v, 3)
    v1 = np.rot90(v1, 3)

    v3 = np.zeros((15,15))

    for i in range(15):
        for j in range(15):
            if v[i][j] != 0:
                v3[i][j] = v[i][j]/v1[i][j]
    

    # fig = plt.figure(200)
    # fig.set_figwidth(12)
    # fig.set_figheight(10)

    b_c_k = sorted(bet_c.items(), key=lambda x:x[1])[::-1]
    # print(b_c_k)
    
    # plt.bar([i for i in range(0,15)], [i[1] for i in b_c_k], alpha=0.3)
    # plt.grid()
    # plt.xticks([i for i in range(15)],[i[0] for i in b_c_k])
    # plt.xlabel("Nodes")
    # plt.ylabel("Betweenness Centrality")


    # fig = plt.figure(300)
    # fig.set_figwidth(12)
    # fig.set_figheight(10)
    
    # sb.set()
    # ax = sb.heatmap(v3, yticklabels=[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], cmap='Blues', annot=True, annot_kws={"size": 7})
    # ax.set(xlabel='Source Nodes', ylabel='Destination Nodes')
    # plt.title("Delivered Bits")

    fig = plt.figure(400)
    fig.set_figwidth(12)
    fig.set_figheight(6)

    res = []

    for i in range(15):
        aux = 0
        for j in range(15):
            aux += v[i][j]
        res.append(aux)
    
    res = res[::-1]

    plt.bar([i for i in range(0,15)], [res[i] for i in [i[0] for i in b_c_k]], alpha=0.3)
    plt.grid()
    plt.xticks([i for i in range(15)],[i[0] for i in b_c_k])
    plt.xlabel("Nodes")
    plt.ylabel("Delivered Bits")

    # fig, ax1 = plt.subplots()
    # fig.set_figwidth(12)
    # fig.set_figheight(6)

    # n_p = num_path(p)
    # print(n_p)

    # len_ps = n_p.keys()
    
    # res = []
    # for i in len_ps:
    #     aux = []
    #     for j in len_path(p, i):
    #         aux.append(v[j[0]][14 - j[-1]])
    #     res.append(mean(aux))
    
    # res1 = []
    # for i in len_ps:
    #     aux = []
    #     for j in len_path(p, i):
    #         aux.append(v1[j[0]][14 - j[-1]])
    #     res1.append(mean(aux))
    
    # plt.bar([i for i in range(1,len(res)+1)], res, alpha=0.3)
    # plt.xlabel("Path Length")
    # plt.ylabel("Average Received bits")
    # plt.grid()

    # ax2 = ax1.twinx()
    # ax2.plot([i for i in range(1,len(res)+1)], [i for i in n_p.values()], '.-', color='red')
    
    # plt.grid()
    # plt.xticks(list(len_ps))
    # plt.ylabel("Number of Paths", color='red')

    # fig = plt.figure(600)
    # fig.set_figwidth(12)
    # fig.set_figheight(10)

    # plt.bar([i for i in range(1,len(res1)+1)], [res[i]/res1[i] for i in range(len(res1))], alpha=0.3)
    # plt.xlabel("Path length")
    # plt.ylabel("Probability of path success")
    # plt.grid()

    plt.show()



if __name__ == '__main__':
    
    #boxplot key rate
    d2 = [
        "sim_2024-01-10_17:38:07",
        "sim_2024-01-10_18:29:31",
        "sim_2024-01-10_09:44:55",
        "sim_2024-01-10_13:04:34",
        "sim_2024-01-10_13:43:34",
        "sim_2024-01-10_22:15:20",
        "sim_2024-01-10_19:09:05",
        "sim_2024-01-10_13:50:16",
        "sim_2024-01-10_21:59:57",
        "sim_2024-01-10_11:39:26",
        "sim_2024-01-10_19:49:49",
        "sim_2024-01-10_17:06:56",
        "sim_2024-01-10_10:11:50",
        "sim_2024-01-10_17:13:05",
        "sim_2024-01-10_16:34:53",
        "sim_2024-01-10_17:31:33",
        "sim_2024-01-10_11:34:54",
        "sim_2024-01-10_10:35:06",
        "sim_2024-01-10_16:53:52",
        "sim_2024-01-10_15:45:28",
        "sim_2024-01-10_11:10:20",
        "sim_2024-01-10_18:49:35",
        "sim_2024-01-10_10:01:52",
        "sim_2024-01-10_14:39:40",
        "sim_2024-01-10_20:30:30",
        "sim_2024-01-10_22:45:33",
        "sim_2024-01-10_10:53:22",
        "sim_2024-01-10_17:25:33",
        "sim_2024-01-10_13:36:03",
        "sim_2024-01-10_20:44:36",
        "sim_2024-01-10_17:44:19",
        "sim_2024-01-10_10:16:53",
        "sim_2024-01-10_14:20:09",
        "sim_2024-01-10_15:52:07",
        "sim_2024-01-10_12:35:44",
        "sim_2024-01-10_19:57:29",
        "sim_2024-01-10_15:13:30",
        "sim_2024-01-10_21:29:15",
        "sim_2024-01-10_11:06:38",
        "sim_2024-01-10_12:54:33",
        "sim_2024-01-10_13:24:49",
        "sim_2024-01-10_18:22:56",
        "sim_2024-01-10_15:40:12",
        "sim_2024-01-10_18:41:44",
        "sim_2024-01-10_16:12:45",
        "sim_2024-01-10_09:57:32",
        "sim_2024-01-10_11:30:16",
        "sim_2024-01-10_10:22:46",
        "sim_2024-01-10_18:56:11",
        "sim_2024-01-10_15:08:10",
        "sim_2024-01-10_15:22:20",
        "sim_2024-01-10_09:52:49",
        "sim_2024-01-10_19:15:32",
        "sim_2024-01-10_17:19:29",
        "sim_2024-01-10_16:41:11",
        "sim_2024-01-10_13:18:49",
        "sim_2024-01-10_14:56:46",
        "sim_2024-01-10_11:01:37",
        "sim_2024-01-10_14:14:15",
        "sim_2024-01-10_13:30:41",
        "sim_2024-01-10_16:47:33",
        "sim_2024-01-10_22:38:11",
        "sim_2024-01-10_22:23:28",
        "sim_2024-01-10_15:34:11",
        "sim_2024-01-10_18:16:22",
        "sim_2024-01-10_17:50:26",
        "sim_2024-01-10_20:57:54",
        "sim_2024-01-10_10:06:53",
        "sim_2024-01-10_21:05:35",
        "sim_2024-01-10_21:51:52",
        "sim_2024-01-10_18:02:59",
        "sim_2024-01-10_12:59:20",
        "sim_2024-01-10_13:55:46",
        "sim_2024-01-10_21:44:35",
        "sim_2024-01-10_22:07:43",
        "sim_2024-01-10_12:48:17",
        "sim_2024-01-10_15:01:56",
        "sim_2024-01-10_21:21:27",
        "sim_2024-01-10_20:37:29",
        "sim_2024-01-10_19:28:36",
        "sim_2024-01-10_19:02:42",
        "sim_2024-01-10_10:39:36",
        "sim_2024-01-10_16:20:37",
        "sim_2024-01-10_19:43:09",
        "sim_2024-01-10_17:56:37",
        "sim_2024-01-10_21:37:12",
        "sim_2024-01-10_14:01:28",
        "sim_2024-01-10_14:33:23",
        "sim_2024-01-10_10:47:13",
        "sim_2024-01-10_14:45:06",
        "sim_2024-01-10_19:36:22",
        "sim_2024-01-10_11:16:38",
        "sim_2024-01-10_15:17:35",
        "sim_2024-01-10_20:17:07",
        "sim_2024-01-10_10:56:20",
        "sim_2024-01-10_20:23:01",
        "sim_2024-01-10_16:00:06",
        "sim_2024-01-10_16:06:43",
        "sim_2024-01-10_15:28:27",
        "sim_2024-01-10_11:20:19",
        "sim_2024-01-10_10:28:36",
        "sim_2024-01-10_12:42:13",
        "sim_2024-01-10_09:50:57",
        "sim_2024-01-10_18:09:50",
        "sim_2024-01-10_11:24:47",
        "sim_2024-01-10_09:39:25",
        "sim_2024-01-10_20:09:54",
        "sim_2024-01-10_13:11:00",
        "sim_2024-01-10_14:27:41",
        "sim_2024-01-10_14:50:35",
        "sim_2024-01-10_21:13:27",
        "sim_2024-01-10_18:35:17",
        "sim_2024-01-10_22:30:42",
        "sim_2024-01-10_19:22:49",
        "sim_2024-01-10_20:50:16",
        "sim_2024-01-10_20:05:10",
        "sim_2024-01-10_14:08:32",
        "sim_2024-01-10_17:00:09",
        "sim_2024-01-10_10:43:36",
        "sim_2024-01-10_16:27:01"
    ]

    #load cresce su 3000, 4000, 5000 m
    d3 = [
        "sim_2024-01-11_11:45:28",
        "sim_2024-01-11_10:46:57",
        "sim_2024-01-11_11:05:53",
        "sim_2024-01-11_10:41:10",
        "sim_2024-01-11_11:12:02",
        "sim_2024-01-11_10:27:44",
        "sim_2024-01-11_11:24:03",
        "sim_2024-01-11_10:35:36",
        "sim_2024-01-11_10:59:42",
        "sim_2024-01-11_10:16:23",
        "sim_2024-01-11_11:17:49",
        "sim_2024-01-11_11:29:51",
        "sim_2024-01-11_10:22:04",
        "sim_2024-01-11_11:37:43",
        "sim_2024-01-11_10:53:23"
    ]

    d4 = [
        'sim_2024-01-27_09:45:42',
        'sim_2024-01-27_10:34:43',
        'sim_2024-01-27_11:19:19',
        'sim_2024-01-27_12:04:54',
        'sim_2024-01-27_12:44:27',
        'sim_2024-01-27_13:34:15',
        'sim_2024-01-27_14:16:42',
        'sim_2024-01-27_15:03:00',
        'sim_2024-01-27_15:47:53',
        'sim_2024-01-27_16:32:47',
        'sim_2024-01-27_17:19:36',
        'sim_2024-01-27_18:06:49',
        'sim_2024-01-27_18:58:05',
        'sim_2024-01-27_19:39:44',
        'sim_2024-01-27_20:25:44',
        'sim_2024-01-27_21:19:23',
        'sim_2024-01-27_22:04:36',
        'sim_2024-01-27_22:56:16',
        'sim_2024-01-27_23:45:41',
        'sim_2024-01-28_00:45:41',
        'sim_2024-01-28_01:28:37',
        'sim_2024-01-28_02:10:20',
        'sim_2024-01-28_03:04:06',
        'sim_2024-01-28_03:50:39',
        'sim_2024-01-28_04:32:39',
        'sim_2024-01-28_05:34:06',
        'sim_2024-01-28_06:23:45',
        'sim_2024-01-28_07:31:20',
        'sim_2024-01-28_08:25:03',
        'sim_2024-01-28_09:24:31',

        'sim_2024-01-23_16:13:30',
        'sim_2024-01-23_17:27:35',
        'sim_2024-01-23_18:34:28',
        'sim_2024-01-23_19:54:07',
        'sim_2024-01-23_21:05:28',
        'sim_2024-01-23_22:16:00',
        'sim_2024-01-23_23:28:20',
        'sim_2024-01-24_00:41:22',
        'sim_2024-01-24_01:50:00',
        'sim_2024-01-24_03:03:02',
        'sim_2024-01-24_04:18:59',
        'sim_2024-01-24_05:31:21',
        'sim_2024-01-24_06:38:55',
        'sim_2024-01-24_07:53:57',
        'sim_2024-01-24_09:07:34',
        'sim_2024-01-24_10:16:53',
        'sim_2024-01-24_11:22:21',
        'sim_2024-01-24_12:31:29',
        'sim_2024-01-24_13:37:16',
        'sim_2024-01-24_14:48:26',
        'sim_2024-01-24_15:58:45',
        'sim_2024-01-24_16:57:08',
        'sim_2024-01-24_18:12:37',
        'sim_2024-01-24_19:33:07',
        'sim_2024-01-24_20:45:49',
        'sim_2024-01-24_22:01:57',
        'sim_2024-01-24_23:11:48',
        'sim_2024-01-24_23:58:10',
        'sim_2024-01-25_01:08:17',
        'sim_2024-01-25_02:20:51'
    ]

    #plot3(d2)
    #plot4(d3)
    #plot_cc("sim_2024-01-13_09:56:16")

    #plot_cc_box_plot(d4)
    #prova(d4)
    #heat_map(d4)
    #plot_prob_new(d4)
    path("sim_2024-02-10_19:18:19")



