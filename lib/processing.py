import os
import glob
import sys

sys.path.append('..')
from multiprocessing import Pool
from multiprocessing import Process

from lib.processing_utils import get_file_list, save_csv, read_txt, video_to_frames, makedir, recut_face_with_landmarks


def generate_path_txt(txt_path, image_path, name_number_dict):
    '''
    生成数据的索引txt,包含path 和 标签
    :param image_path: 包含多个子文件夹,每个子文件夹的名字是类别名字,子文件夹下包含这个类别的数据
    :name_number_dict: 文件名到数字编号的映射字典
    :return:
    '''
    class_list = os.listdir(image_path)
    for class_index in class_list:
        class_number = name_number_dict[class_index]
        class_path = os.path.join(image_path, class_index)
        class_image_path_list = get_file_list(class_path)
        for path in class_image_path_list:
            with open(txt_path, 'a+') as f:
                f.write(path)
                f.write(' ')
                f.write(path)
                f.write(' ')
                f.write(path)
                f.write(' ')
                f.write(str(class_number))
                f.write('\n')


def generate_cefa_path_txt(txt_path, image_path):
    '''
    为多模态数据集生成txt索引
    :param txt_path:
    :param image_path:
    :param name_number_dict:
    :return:
    '''
    # file path: /home/data/shicaiwei/cefa/CeFA-Race/AF/AF-415/1_415_1_1_1/ir
    # image_path: /home/data/shicaiwei/cefa/CeFA-Race
    race_list = os.listdir(image_path)
    for race in race_list:
        race_dir = os.path.join(image_path, race)  # /home/data/shicaiwei/cefa/CeFA-Race/AF
        subject_list = os.listdir(race_dir)
        for subject in subject_list:
            subject_dir = os.path.join(race_dir, subject)  # /home/data/shicaiwei/cefa/CeFA-Race/AF/AF-415
            scene_list = os.listdir(subject_dir)
            for scene in scene_list:
                label = int(scene.split('_')[-1])
                label = 1 if label == 1 else 0
                scene_dir = os.path.join(subject_dir,
                                         scene)  # /home/data/shicaiwei/cefa/CeFA-Race/AF/AF-415/1_415_1_1_1

                rgb_dir = os.path.join(scene_dir, 'profile')
                rgb_path_list = get_file_list(rgb_dir).sort()
                ir_dir = os.path.join(scene_dir, 'ir')
                ir_path_list = get_file_list(ir_dir).sort()
                depth_dir = os.path.join(scene, 'depth')
                depth_path_list = get_file_list(depth_dir).sort()

                image_number = len(rgb_path_list)
                for index in range(image_number):
                    with open(txt_path, 'a+') as f:
                        f.write(rgb_path_list[index])
                        f.write(' ')
                        f.write(depth_path_list[index])
                        f.write(' ')
                        f.write(ir_path_list[index])
                        f.write(' ')
                        f.write(str(label))
                        f.write('\n')


def generate_msu_fasd(msu_path):
    '''
    提取msu_fasd 数据集的视频文件到帧图像,并按照train_list 和test_list划分数据集.
    官方文件只有train_sub_list.txt 和test_sub_list.txt, 这里去test_sub_list.txt 之外的其他样本视频为train_all.txt
    :param msu_path:
    :return:
    '''

    def extract_data_via_number(number, train):
        video_name = "*client0" + str(number) + "*.avi"
        video_path = os.path.join(msu_path, video_name)
        video_path_list = glob.glob(video_path)
        print(video_path_list)
        for path in video_path_list:
            video_name = path.split('/')[-1]
            class_name = video_name.split('_')[0]
            if class_name == 'real':
                device = video_name.split('_')[-3]
                if train:
                    save_path = os.path.join(msu_path, 'MSU_FASD', 'train/living', str(number), device)
                else:
                    save_path = os.path.join(msu_path, 'MSU_FASD', 'test/living', str(number), device)
                if not os.path.exists(save_path):
                    makedir(save_path)
                print(save_path)
                # video_to_frames(path, pathOut=save_path)
            else:
                device = video_name.split('_')[-5]
                spoofing_type = video_name.split('_')[-3]
                if train:
                    save_path = os.path.join(msu_path, 'MSU_FASD', 'train/spoofing', str(number), device, spoofing_type)
                else:
                    save_path = os.path.join(msu_path, 'MSU_FASD', 'test/spoofing', str(number), device, spoofing_type)
                if not os.path.exists(save_path):
                    makedir(save_path)
                print(save_path)
                # video_to_frames(path, pathOut=save_path)

    train_txt = os.path.join(msu_path, 'train_all.txt')
    test_txt = os.path.join(msu_path, 'test_sub_list.txt')

    train_sample_labels = read_txt(train_txt)
    for index in train_sample_labels:
        print("processing train", index)
        extract_data_via_number(index, train=True)

    test_sample_labels = read_txt(test_txt)
    for index in test_sample_labels:
        print("processing test", index)
        extract_data_via_number(index, train=False)


import cv2


def resize_face_img():
    data_path = "/home/data/shicaiwei/oulu/Test_face"
    save_dir = "/home/data/shicaiwei/oulu/Test_face_normal"
    video_list = os.listdir(data_path)
    video_list.sort()
    for video in video_list:
        video_path = os.path.join(data_path, video)
        img_path_list = get_file_list(video_path)
        img_path_list.sort()

        save_dir = os.path.join(save_dir, video)
        makedir(save_dir)

        for path in img_path_list:
            print(path)
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img_name = path.split('/')[-1]
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img)

        save_dir = "/home/data/shicaiwei/oulu/Test_face_normal"


def get_landmarks_face(start, end):
    data_path = "/home/data/shicaiwei/oulu/Test_face_normal"
    save_dir = "/home/data/shicaiwei/oulu/Test_face_landmarks"
    video_list = os.listdir(data_path)
    video_list.sort()
    video_list = video_list[start:end]
    print(start)
    for video in video_list:
        video_path = os.path.join(data_path, video)
        img_path_list = get_file_list(video_path)
        img_path_list.sort()

        save_dir = os.path.join(save_dir, video)
        makedir(save_dir)
        for path in img_path_list:
            print(path)
            img = cv2.imread(path)
            landmarks_face = recut_face_with_landmarks(img)
            img_name = path.split('/')[-1]
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, landmarks_face)

        save_dir = "/home/data/shicaiwei/oulu/Test_face_landmarks"


if __name__ == '__main__':
    # msu_path = "/home/bbb//shicaiwei/data/liveness_data/msu_mfsd"
    # generate_msu_fasd(msu_path=msu_path)

    process_list = []
    for i in range(20):  # 开启5个子进程执行fun1函数
        p = Process(target=get_landmarks_face, args=(i * 90, i * 90 + 90,))  # 实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    # get_landmarks_face(1, 10000)
