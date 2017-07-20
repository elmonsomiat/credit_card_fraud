
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sys
from PIL import Image
import os, errno
import csv



def vectorize_categories(categories, y_all):
    '''
    categories: all the individual catogories y_all contains
    y_all: categories of all the data
    
    Convert the list of categories y_all for each picture
    and convert them into a binary vector for each of the 
    corresponding categories
    e.g. ['agriculture, haze'] --> [1,0,0,1,0,...,0]
    '''
    
    categories_dic = {}
    for i in range(len(categories)):
        categories_dic.update({categories[i]: i})
        
    df = pd.DataFrame(y_all.str.split(' ').tolist()) #separate categories   
    y_all_array = df.as_matrix()

    #drop all none-entries from the created array
    y_all_list = []
    for i in range(len(y_all_array)):
        y_all_list.append(y_all_array[i][y_all_array[i] != np.array(None)].tolist()) 
        
    #apply the dictionary to the categories
    y_all_list_vec = []
    for i in range(len(y_all_list)):    
        y_all_list_vec.append((np.vectorize(categories_dic.get)(np.array(y_all_list[i]))).tolist())
    
    #binarize the vector    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_vec_all = pd.DataFrame(mlb.fit_transform(y_all_list_vec))
    
    return y_vec_all



def image_to_df(path, files, newshape,RGB):
    '''
    path: path where the images are contained
    files: name of the images which have to be converted to df
    newshape: new shape of the pictures (width, height)
    RGB: 0 for red, 1 for green and 2 for blue     
    
    Convert the list of categories y_all for each picture
    and convert them into a binary vector for each of the 
    corresponding categories and store as Pandas DataFrame
    e.g. ['agriculture, haze'] --> [1,0,0,1,0,...,0]
    '''
    j = 0
    for i in files:
        im = Image.open(path+i) #read image one by one
        im = im.resize(newshape,Image.ANTIALIAS)
        pixels_flat = np.array(im)[:,:,RGB].flatten() #convert pixels to a vector 
        im.close()
        if i == files[0]:
            pixel_array = [pixels_flat]
        else:
            pixel_array = np.vstack((pixel_array,[pixels_flat]))
        
        if j%100==0:
            print('.', end='', flush=True)
            if j%1000==0 and j!=0:
                print(j, flush=True)
        j+=1
    print('end')        

    pixel_df = pd.DataFrame(pixel_array,index=files)
    pixel_df['image_name'] = pixel_df.index
    pixel_df['image_name'] = pixel_df['image_name'].apply(lambda x: x.split('.')[0])
    pixel_df = pixel_df.reset_index(drop=True)
    del(pixel_array)
    
    return pixel_df





def write_pixels_to_file(outFile, path, files, newshape,RGB):
    '''
    outFile: name of the output .csv file
    path: path where the images are contained
    files: name of the images which have to be saved as pixels
    newshape: new shape of the pictures (width, height)
    RGB: 0 for red, 1 for green and 2 for blue  
    
    Convert the list of categories y_all for each picture
    and convert them into a binary vector for each of the 
    corresponding categories and store as Pandas DataFrame
    e.g. ['agriculture, haze'] --> [1,0,0,1,0,...,0]
    '''
    j = 0
    try:
        os.remove(outFile)
    except OSError:
        pass
    
    for i in files: 
        with open(outFile, 'a') as csvfile:
            writer=csv.writer(csvfile, delimiter=',',lineterminator='\n')
            if j == 0:
                writer.writerow(list(range(newshape[0]*newshape[1]+1)))
                
            im = Image.open(path+i) #read image one by one
            im = im.resize(newshape,Image.ANTIALIAS)
            pixels_flat = np.array(im)[:,:,RGB].flatten() #convert pixels to a vector 
            im.close()
            pixels_flat_name = i.split('.')[0]
            writer.writerow(np.append(pixels_flat, [pixels_flat_name]))
            
            if j%100==0:
                print('.', end='', flush=True)
                if j%1000==0 and j!=0:
                    print(j, flush=True)
        j+=1
    print('end') 
    return


   
def append_rotated_images(X, y, rotNum):
    '''
    X: pixels of the input image
    y: binarised categories for this picture (e.g.: [1,0,1,0,0...])
    rotNum: how many times the image has to be rotated
    
    Rotate the input images rotNum of times and store the pixels
    '''
    pixels_array = X
    newlen = int(np.sqrt(len(pixels_array)))
    pixels_array_reshape = np.reshape(pixels_array,(newlen, newlen))
    im = Image.fromarray(np.uint8(pixels_array_reshape))
    y_rot_all = pd.DataFrame(columns=list(range(len(y))))
    for i in range(rotNum):
        
        img_rot = im.rotate((360*(i+1)/(rotNum)), expand=True) #rotate image
       
        rot_angle = np.pi*(i+1)*2/(rotNum)
        if abs(np.cos(rot_angle))>= abs(np.cos(np.pi/4)):
            expand_size = int(newlen*(abs(np.cos(rot_angle))+abs(np.sin(rot_angle)))**2)+1
        else:
            rot_angle = np.pi/2-rot_angle
            expand_size = int(newlen*(abs(np.cos(rot_angle))+abs(np.sin(rot_angle)))**2)+1
            
        
        img_rot = img_rot.resize((expand_size, expand_size),Image.ANTIALIAS) #resize to intial size        
        img_rot = img_rot.crop((expand_size/2 - newlen/2,\
                              expand_size/2 - newlen/2,\
                              expand_size/2 + newlen/2,\
                              expand_size/2 + newlen/2)) #crop image to delete black parts
        
        img_rot = img_rot.resize((newlen, newlen),Image.ANTIALIAS) #resize image(just in case!)
        
        pixels_flat = np.array(img_rot)[:,:].astype(np.int32).flatten()  #store pixels
        
        if i == 0:
            pixel_array = [pixels_flat]
        else:
            pixel_array = np.vstack((pixel_array,[pixels_flat]))
            
        pixels_rot_df = pd.DataFrame(pixel_array, columns=list(X))
        y_rot_all = y_rot_all.append(pd.DataFrame([y])).astype(int)
    X_rot_all = pd.DataFrame(pixels_rot_df)
    
    return X_rot_all, y_rot_all





def softmax_mine(y):
    '''
    softmax function for y
    '''
    for i in range(len(y)):
        if y[i][0]>y[i][1]:
            y[i][0]=1
            y[i][1]=0
        else:
            y[i][1]=1
            y[i][0]=0
    return y
        