import os
import random
import shutil
from PIL import Image
from collections import defaultdict


def split_images_vaihingen(src_folder, src_folder_mask, dst_folder):
    folders = ['train','val','test']
    for folder in folders:
        os.makedirs(os.path.join(dst_folder, folder, 'image'))
        os.makedirs(os.path.join(dst_folder, folder, 'mask_pseudo'))

    train_folder = os.path.join(dst_folder, 'train')
    val_folder = os.path.join(dst_folder, 'val')
    test_folder = os.path.join(dst_folder, 'test')

    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    prefixes = []
    for f in image_files:
        n = f.replace('.jpg', '').split('_')[0]
        if n not in prefixes:
            prefixes.append(n)
    random.shuffle(prefixes)
    train_size = int(len(prefixes) * 0.7)
    val_size = int(len(prefixes) * 0.2) #vaihingen 不需要这个
    test_size = int(len(prefixes) * 0.3)

    train_files = prefixes[:train_size]
    if 'vaihingen' in src_folder:
        val_files = prefixes[train_size:train_size + int(test_size*0.2)]
        val_size = 0
    else:
        val_files = prefixes[train_size:train_size + val_size]
    test_files = prefixes[train_size + val_size:]

    r = ['hflip','vflip','digflip']
    for i in train_files:
        shutil.copy(os.path.join(src_folder, i+'.jpg'), os.path.join(train_folder, 'image', i+'.jpg'))
        shutil.copy(os.path.join(src_folder_mask, i+'.png'),
                    os.path.join(train_folder, 'mask_pseudo', i+'.png'))
        for j in r:
            shutil.copy(os.path.join(src_folder, i +'_'+ j+'.jpg'), os.path.join(train_folder, 'image', i+'_'+ j + '.jpg'))
            shutil.copy(os.path.join(src_folder_mask, i+'_'+ j + '.png'),
                        os.path.join(train_folder, 'mask_pseudo', i+'_'+ j + '.png'))
    for i in val_files:
        shutil.copy(os.path.join(src_folder, i + '.jpg'), os.path.join(val_folder, 'image', i + '.jpg'))
        shutil.copy(os.path.join(src_folder_mask, i + '.png'),
                    os.path.join(val_folder, 'mask_pseudo', i + '.png'))
        for j in r:
            shutil.copy(os.path.join(src_folder, i + '_' + j + '.jpg'),
                        os.path.join(val_folder, 'image', i + '_' + j + '.jpg'))
            shutil.copy(os.path.join(src_folder_mask, i + '_' + j + '.png'),
                        os.path.join(val_folder, 'mask_pseudo', i + '_' + j + '.png'))
    for i in test_files:
        shutil.copy(os.path.join(src_folder, i + '.jpg'), os.path.join(test_folder, 'image', i + '.jpg'))
        shutil.copy(os.path.join(src_folder_mask, i + '.png'),
                    os.path.join(test_folder, 'mask_pseudo', i + '.png'))
        for j in r:
            shutil.copy(os.path.join(src_folder, i + '_' + j + '.jpg'),os.path.join(test_folder, 'image', i + '_' + j + '.jpg'))
            shutil.copy(os.path.join(src_folder_mask, i + '_' + j + '.png'),
                        os.path.join(test_folder, 'mask_pseudo', i + '_' + j + '.png'))


def split_images(src_folder, src_folder_mask, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    folders = ['train', 'val', 'test']
    ratios = [0.6, 0.03, 0.37]

    for folder in folders:
        os.makedirs(os.path.join(dst_folder, folder, 'image'))
        os.makedirs(os.path.join(dst_folder, folder, 'mask_pseudo'))

    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    start_idx = 0
    for i, folder in enumerate(folders):
        end_idx = int(start_idx + len(image_files) * ratios[i])
        for j in range(start_idx, end_idx):
            src_path = os.path.join(src_folder, image_files[j])
            src_path_mask = os.path.join(src_folder_mask, image_files[j].replace('.jpg', '.png'))
            dst_path = os.path.join(dst_folder, folder, 'image', image_files[j])
            dst_path_mask = os.path.join(dst_folder, folder, 'mask_pseudo', image_files[j].replace('.jpg', '.png'))
            shutil.copy(src_path, dst_path)
            shutil.copy(src_path_mask, dst_path_mask)
        start_idx = end_idx


src_folder = '/media/oyasumi/SSD256/seg_datasets/new_potsdam/all_aug/images'
src_folder_mask = '/media/oyasumi/SSD256/seg_datasets/new_potsdam/all_aug/labels'
dst_folder = '/media/oyasumi/SSD256/seg_datasets/new_potsdam/6_0.3_3.7'
if 'vaihingen' in src_folder:
    split_images_vaihingen(src_folder, src_folder_mask, dst_folder)
else:
    split_images(src_folder, src_folder_mask, dst_folder)
