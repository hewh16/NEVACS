import os, h5py, cv2
import numpy as np

train_path = './cell-dataset/train/'
test_path = './cell-dataset/test/'

train_type_list = os.listdir(train_path)
train_type_list.sort()
test_type_list = os.listdir(test_path)
test_type_list.sort()
print(train_type_list)

# shape = 

train_data = []
test_data = []

for train_type in train_type_list:
    data_list = os.listdir(train_path+train_type)
    data_list.sort()
    data_idx = []
    print(data_list)
    for data in data_list:
        img_list = os.listdir(train_path+train_type+'/'+data)
        data_group = []
        for img in img_list:
            if '.jpg' not in img:
                continue
            img_data = cv2.imread(train_path+train_type+'/'+data+'/'+img)
            img_data = np.array([img_data[:,:,0]])
            img_data[img_data!=0] = 1
            # img_data = np.zeros((120,120,2),dtype='int')
            
            #print(img_data.shape)
            #print(img_data[img_data==125])
            
            data_group.append(img_data)
        #print(data_group.shape)
        data_group = np.array(data_group)
        #print(data_group.shape)
        #exit()
        data_idx.append(data_group)
    train_data.append(data_idx)
np.save('./cell-dataset/train.npy',train_data)

for test_type in test_type_list:
    data_list = os.listdir(test_path+test_type)
    data_list.sort()
    data_idx = []
    print(data_list)
    for data in data_list:
        img_list = os.listdir(test_path+test_type+'/'+data)
        data_group = []
        for img in img_list:
            if '.jpg' not in img:
                continue
            img_data = cv2.imread(test_path+test_type+'/'+data+'/'+img)
            img_data = np.array([img_data[:,:,0]])
            img_data[img_data!=0] = 1
            #print(img_data.shape)
            data_group.append(img_data)
        #print(data_group.shape)
        data_group = np.array(data_group)
        print(data_group.shape)
        #exit()
        data_idx.append(data_group)
    test_data.append(data_idx)
np.save('./cell-dataset/test.npy',test_data)

