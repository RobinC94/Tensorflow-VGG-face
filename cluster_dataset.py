import os
import json
import shutil

output_folder = './output/'
image_folder = '/data1/datasets/PeopleInfo/'
json_path = './cluster_result.json'

if __name__ == "__main__":

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with open('cluster_result.json', 'r') as f:
        face_dict = json.load(f)
        for key, dir_list in face_dict.items():
            class_path = output_folder + "HJFEAT_" + str(key)
            print(class_path)
            if not os.path.isdir(class_path):
                os.makedirs(class_path)
            for dir in dir_list:
                dir_src_path = image_folder + str(dir) + "/Screensave/"
                print(dir_src_path)
                if not os.path.isdir(dir_src_path):
                    raise ValueError("f**k!")
                for image in os.listdir(dir_src_path):
                    image_path = dir_src_path + image
                    if os.path.splitext(image_path)[-1] == ".bmp":
                        shutil.copy(image_path, class_path)



