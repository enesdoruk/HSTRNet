import os 
import cv2
import argparse
import shutil
import numpy as np
from tqdm import tqdm



def splitData(data_root, new_dataset_root):
    lr = []
    gt = []
    ref = []


    os.makedirs(new_dataset_root, exist_ok=True)

    os.makedirs('{}'.format(new_dataset_root + '/' + 'sequences' + '/' + 'gt'), exist_ok=True)
    os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' +  'ref'), exist_ok=True)
    os.makedirs('{}'.format(new_dataset_root + '/' + 'sequences' + '/' + 'lr'), exist_ok=True)

    sequences = os.listdir(data_root + '/' + 'sequences')
    sequences.sort()

    for seq in sequences:
        os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'gt' + '/' + seq) , exist_ok=True)
        os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'ref' + '/' + seq) , exist_ok=True)
        os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'lr' + '/' + seq) , exist_ok=True)

        img_folder = os.listdir(data_root + '/' + 'sequences' + '/' + seq)
        img_folder.sort()

        for i in tqdm(range(len(img_folder)), desc= "{}/images".format(seq)):
            os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'gt' + '/' + seq + '/' + img_folder[i]) , exist_ok=True)
            os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'ref' + '/' + seq + '/' + img_folder[i]) , exist_ok=True)
            os.makedirs('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'lr' + '/' + seq + '/' + img_folder[i]) , exist_ok=True)

            images = os.listdir('{}'.format(data_root + '/' +  'sequences' + '/' + '/' + seq + '/' + img_folder[i]))
            images.sort()
            

            shutil.copy('{}'.format(data_root + '/' +  'sequences'  + '/' + seq + '/' + img_folder[i] + '/' + images[2]),
                        '{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'gt' + '/' + seq + '/' + img_folder[i]))
            gt.append('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'gt' + '/' + seq + '/' + img_folder[i] + '/' + images[2]))
            gt.sort()

            
            lr_image = cv2.imread(data_root + '/' +  'sequences'  + '/' + seq + '/' + img_folder[i] + '/' + images[2])
            lr_downsample = cv2.resize(lr_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            lr_upsample = cv2.resize(lr_downsample, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'lr' + '/' + seq + '/' + img_folder[i] + '/' + images[2]), lr_upsample)

            lr.append('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'lr' + '/' + seq + '/' + img_folder[i] + '/' + images[2]))
            lr.sort()

            shutil.copy('{}'.format(data_root + '/' +  'sequences'  + '/' + seq + '/' + img_folder[i] + '/' + images[1]),
                        '{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'ref' + '/' + seq + '/' + img_folder[i]))
            
            ref.append('{}'.format(new_dataset_root + '/' +  'sequences' + '/' + 'ref' + '/' + seq + '/' + img_folder[i] + '/' + images[1]))
            ref.sort
            
        continue

    train_ratio = 0.8

    train_lr_txt = []
    test_lr_txt = []

    train_ref_txt = []
    test_ref_txt = []

    train_gt_txt = []
    test_gt_txt = []



    train_lr_txt = lr[:int(len(lr)*train_ratio)]
    test_lr_txt = lr[int(len(lr)*train_ratio):]

    train_ref_txt = ref[:int(len(ref)*train_ratio)]
    test_ref_txt = ref[int(len(ref)*train_ratio):]

    train_gt_txt = gt[:int(len(gt)*train_ratio)]
    test_gt_txt = gt[int(len(gt)*train_ratio):]

    with open(f'{new_dataset_root}/train_lr.txt', 'w') as f:
        for line in train_lr_txt:
            f.write(f"{line}\n")
        print("Done creating train low resolution split data")

    with open(f'{new_dataset_root}/test_lr.txt', 'w') as f:
        for line in test_lr_txt:
            f.write(f"{line}\n")
        print("Done creating test low resolution split data")

    
    with open(f'{new_dataset_root}/train_ref.txt', 'w') as f:
        for line in train_ref_txt:
            f.write(f"{line}\n")
        print("Done creating train reference split data")

    with open(f'{new_dataset_root}/test_ref.txt', 'w') as f:
        for line in test_ref_txt:
            f.write(f"{line}\n")
        print("Done creating test reference split data")


    with open(f'{new_dataset_root}/train_gt.txt', 'w') as f:
        for line in train_gt_txt:
            f.write(f"{line}\n")
        print("Done creating train gt split data")

    with open(f'{new_dataset_root}/test_gt.txt', 'w') as f:
        for line in test_gt_txt:
            f.write(f"{line}\n")
        print("Done creating test gt split data")




def main(data_root, new_dataset_root):
    splitData(data_root, new_dataset_root)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="splitting dataset")
    parser.add_argument("--data_root", default='/home/adastec/vimeo_triplet_full', type=str)
    parser.add_argument("--new_data_root", default='/home/adastec/vimeo_new', type=str)
    args = parser.parse_args()

    main(args.data_root, args.new_data_root)

