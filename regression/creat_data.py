import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
import time

def clean_file(data_path,clean_data_path):
    clean_data = open(clean_data_path, "w")
    fo = open(data_path, "r")
    for line in fo.readlines():
        line = line.split(",")
        count = int(line[5]) + int(line[6])
        clean_data.writelines(str(line[3][:10]) + "," + str(count) + "\n")

def create_time(data_time_file):
    data_time_file = open(data_time_file, "w")

    box = []
    for x in range(0,13):
        if x == 1 or x == 3 or x == 5 or x == 7 or x == 8 or x == 10 or x == 12:
            for y in range(1,32):
                for z in range(0,24):
                    data_line = 2018000000 + z + y*100 + x*10000
                    data_time_file.writelines(str(data_line)+","+str(0) + "\n")
                    box.append([data_line, 0])
        if x == 4 or x == 6 or x == 9 or x == 11:
            for y in range(1,31):
                for z in range(0,24):
                    data_line = 2018000000 + z + y*100 + x*10000
                    data_time_file.writelines(str(data_line)+","+str(0) + "\n")
                    box.append([data_line, 0])
        if x ==2:
            for y in range(1,29):
                for z in range(0,24):
                    data_line = 2018000000 + z + y*100 + x*10000
                    data_time_file.writelines(str(data_line)+","+str(0) + "\n")
                    box.append([data_line, 0])
    data_time_file.close()



def final_file(final_file_path):
    data_time_file = "./data_time_file.txt"
    data_all = "./clean_data.txt"
    final_file = open(final_file_path, "w")
    fo_all = open(data_all, "r")
    fo_time = open(data_time_file, "r")
    num = 0
    line_all_box = []
    for line_all in fo_time.readlines():
        line_all = line_all.split(",")
        line_all_box.append([line_all[0], int(line_all[1])])
    line_box = []
    for line in fo_all.readlines():
        line = line.split(",")
        line_box.append([line[0], int(line[1])])

    for i in range(len(line_all_box)):
        b = []
        T1 = time.time()
        for j in range(len(line_box)):
            if line_all_box[i][0] == line_box[j][0]:
                b.append(line_box[j][1])
            # if line_all_box[i][0] != line_box[j][0] and line_all_box[i][0] == line_box[j-1][0]:
            # break
        T2 = time.time()
        t = (T2 - T1)

        final_file.writelines(str(num) + "," + str(sum(b)) + "\n")
        num = num + 1
        print(num)


if __name__ == '__main__':
    data_path = './data.txt'
    clean_data_path = "./clean_data.txt"  # 整理好的文件，只有时间和人数
    data_time_file = "./data_time_file.txt"  #全部时间的文件
    final_file_path = "./final_file.txt"  #最后生成的文件
    clean_file(data_path, clean_data_path)  #整理文件
    create_time(data_time_file)  #创建时间文件
    final_file(final_file_path)  #得到最后的清洗好的文件
