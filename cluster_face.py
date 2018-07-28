import os
import numpy as np
from Bio.Cluster import clusterdistance
import json

def get_dis_of_face(face1, face2):
    data = np.vstack((face1,face2))
    l1 = np.shape(face1)[0]
    l2 = np.shape(face2)[0]
    index1 = range(l1)
    index2 = range(l1, l1 + l2, 1)

    dis = clusterdistance(data, index1=index1, index2=index2, dist='e', method='v')
    return dis

def find_min_dis(dis_mat, index_list):
    index_num = len(index_list)
    min_i = -1
    min_j = -1
    min_dis = 10000
    for i in range(index_num-1):
        for j in range(i+1, index_num, 1):
            index_i = index_list[i]
            index_j = index_list[j]
            dis = dis_mat[index_i][index_j]
            if dis < min_dis:
                min_dis = dis
                min_i = index_i
                min_j = index_j
    return min_i, min_j


def update_dis_mat(dis_mat, face_dict, i):

    face_i = get_face_array(face_dict[i])
    for j in face_dict.keys():
        if i == j:
            continue
        face_j = get_face_array(face_dict[j])
        dis = get_dis_of_face(face_i, face_j)
        if j < i:
            dis_mat[j][i] = dis
        elif j > i:
            dis_mat[i][j] = dis

def get_face_array(data_list):
    face_list = []
    for data in data_list:
        file_path = './data/' + data
        face = np.load(file_path)
        face_list.append(face)

    return np.vstack(face_list)


if __name__ == "__main__":
    face_list = []
    face_dict = {}
    data_path = './data/'
    data_file_list = os.listdir(data_path)
    file_num = len(data_file_list)
    for file in data_file_list:
        file_path = data_path + file
        face_data = np.load(file_path)
        face_list.append(face_data)

    dis_mat = [[10000 for x in range(file_num)] for y in range(file_num)]
    for i in range(file_num):
        face_dict[i] = [data_file_list[i]]
        for j in range(i+1, file_num, 1):
            face1 = face_list[i]
            face2 = face_list[j]
            dis = get_dis_of_face(face1, face2)
            dis_mat[i][j] = dis

    while(len(face_dict.keys()) > 101):
        index_list = face_dict.keys()
        i,j = find_min_dis(dis_mat, index_list)
        print(str(len(index_list)) + ':', i, j)
        face_dict[i] += face_dict[j]
        face_dict.pop(j)
        update_dis_mat(dis_mat,face_dict,i)

    with open('cluster_result.json', 'w') as f:
        json.dump(face_dict, f)



    print(face_dict)
