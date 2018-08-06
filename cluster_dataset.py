import os
import json

if __name__ == "__main__":
    output_folder = './output/'
    image_folder = '/data1/datasets/PeopleInfo/'
    json_path = './cluster_result.json'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with open('cluster_result.json', 'r') as f:
        face_dict = json.load(f)
        print type(face_dict)
        for key, dir_list in face_dict.items():
            print key, dir_list


