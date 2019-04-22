
# coding: utf-8

# In[16]:


import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
###
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
# from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.nasnet import preprocess_input
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
import time


feat_models_dic = {'InceptionV3': ['avg_pool', 229, 2048],
                   'Xception': ['avg_pool', 229, 2048],
                   'VGG16': ['fc2', 224, 4096],
                   'VGG19': ['fc2', 224, 4096],
                   'MobileNetV2': ['global_average_pooling2d_1', 224, 1280],
                   'NASNetMobile': ['global_average_pooling2d_2', 224, 1056],
                   'DenseNet121': ['fc1000', 224, 1000]
                   }

def feat_data(baseModel = 'InceptionV3', imageFormat = 'jpeg'):
    data_dir = Path('./www')
    labeled_dir = data_dir / 'LABELED'
    unlabeled_dir = data_dir / 'UNLABELED'
    normal_cases_dir = labeled_dir / 'NORMAL'
    abnormal_cases_dir = labeled_dir / 'ABNORMAL'
    normal_cases = normal_cases_dir.glob('*.' + imageFormat)
    abnormal_cases = abnormal_cases_dir.glob('*.' + imageFormat)
    unlabeled_cases = unlabeled_dir.glob('*.' + imageFormat)
    data = []

    # Go through all the normal cases. The label for these cases will be 0
    for img in normal_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img,0))

    # Go through all the abnormal cases. The label for these cases will be 1
    for img in abnormal_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img, 1))

    # Go through all the unlabeled cases. The label for these cases will be 'NA'
    for img in unlabeled_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img, np.nan))    

    # Get a pandas dataframe from the data we have in our list 
    data = pd.DataFrame(data, columns=['image', 'label'],index=None)

    # Shuffle the data 
    data = data.sample(frac=1.0).reset_index(drop=True)
    
    start = time.time()
    
    if baseModel == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet')
    elif baseModel == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet')
    elif baseModel == 'Xception':
        base_model = Xception(weights='imagenet')
    elif baseModel == 'VGG16':
        base_model = VGG16(weights='imagenet')
    elif baseModel == 'VGG19':
        base_model = VGG19(weights='imagenet')
    elif baseModel == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet')
    elif baseModel == 'NASNetMobile':
        base_model = NASNetMobile(weights='imagenet')
    else:
        raise Exception('{} is not a valid featurization model, please choose another one!'.format(baseModel))
        
    
    
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(feat_models_dic[baseModel][0]).output)
    fc_features =np.zeros((data.shape[0],feat_models_dic[baseModel][2]))
    for i in range(data.shape[0]):
        img_path = str(data['image'][i])
        img = image.load_img(img_path, target_size=(feat_models_dic[baseModel][1], feat_models_dic[baseModel][1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc_features[i]=model.predict(x)

    df = pd.DataFrame(fc_features, columns=['X'+str(i) for i in range(fc_features.shape[1])],index=None)
    final_data = pd.concat([df, data], axis= 1)
    final_data['image'] = final_data['image'].astype(str)
    final_data.to_csv('final_data_test.csv')
    
    end = time.time()
    
    print(end-start)
    return final_data

if __name__ == '__main__':

    print('Time for VGG19')
    feat_data(baseModel = 'VGG19')
    print('Time for DenseNet121')
    feat_data(baseModel = 'DenseNet121')
    print('Time for InceptionV3')
    feat_data(baseModel = 'InceptionV3')
    print('Time for Xception')
    feat_data(baseModel = 'Xception')