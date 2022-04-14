'''
This script takes an image input ,preprocess it and returns a prediction according to the ml model

'''


import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib.patches import Rectangle
from io import BytesIO
from skimage.feature.peak import peak_local_max
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
import scipy
import os
import re
import base64
import tempfile
import pydicom as dicom

model=tf.keras.models.load_model('model.hdf5')
print("Loaded model")
import boto3
import contextlib
import sys
# from botocore.exceptions import NoCredentialsError


aws_access_key_id='AKIARPEY5DNLOXYZDLFZ'
aws_secret_access_key='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'

@contextlib.contextmanager
def s3connection(key, secret):
    print("Connecting to S3")
    s3 = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=secret)
    print("Established Connection")
    try:
        yield s3
    except Exception as e:
        print("Error occurred: ".format(e))
        sys.exit(1)
    # except NoCredentialsError:
    #     print("Credentials not available")
    #     sys.exit(1)

    finally:
        print("Keeping Connection")
        

with s3connection(aws_access_key_id, aws_secret_access_key) as s3:
    print("success")
def img_to_base64_str(img):
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    buffered.seek(0)
    img_bytes=buffered.getvalue()
    img_str="data:image/png;base64,"+base64.b64encode(img_bytes).decode()
    
    return img_str


def preprocess_image(image):
    '''
    This function preprocesses the image

    Args:
        image: str 
            image to be preprocessed

    Returns:
        image: numpy array
            preprocessed image

    raises:
        Exception: if image is not found
    
    '''
    test_image=[]
    
        # image=image.resize((224,224))
        # image=tf.keras.preprocessing.image.img_to_array(image)
        # image=image/255
        # image=image.reshape((1,224,224,3))
    # image=cv2.imread(image)
    image=dicom.dcmread(image)
    # image.PhotometricInterpretation = 'RGB'
    # image.SamplesPerPixel = 3
    # image.BitsAllocated = 8
    # image.BitsStored = 8
    # image.HighBit = 7
    # image.add_new(0x00280006, 'US', 0)
    image=image.pixel_array
    cv2.imwrite('test.png',image)
    image=cv2.imread('test.png')

    image=Image.fromarray(image,'RGB')
    image=image.resize((256,256))
    test_image.append(np.array(image))
    test_image=np.array(test_image)
    test_image=test_image/255
    
    return test_image


def plot_heatmap(img,inspection_code):
  
    # pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(prediction)
    #Get weights for all classes from the prediction layer
    last_layer_weights = model.layers[-1].get_weights()[0] #Prediction layer
   
    #Get weights for the predicted class.
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    
    #Get output from the last conv. layer
    last_conv_model = Model(model.input, model.get_layer("conv5_block16_concat").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
  
    
    #Upsample/resize the last conv. output to same size as original image
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
   
    
    
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 1024)), 
                 last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    
 
    heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0
    
    #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    #with rel threshold of 0.5 (compared to the max peak). 
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10) 

    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(heat_map, cmap='jet', alpha=0.30)
    for i in range(0,peak_coords.shape[0]):
       
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-30, y-30), 80,80,linewidth=1,edgecolor='r',facecolor='none'))
    plt.savefig(inspection_code+'.png')
    
    
    image = cv2.imread(inspection_code+".png")
    s3.upload_file(Filename=inspection_code+".png",Bucket="chest-predictions",Key=inspection_code+".png")
    image=Image.fromarray(image,'RGB')
    image=image.resize((256,256))
    
    #return single image
    
    return img_to_base64_str(image)

def predict_base64_image(name, contents,inspection_code):
    fd, file_path = tempfile.mkstemp()
    with open(fd,'wb') as f:
        f.write(base64.b64decode(contents))
    

    image=preprocess_image(file_path)
    prediction=model.predict(image)
    prediction_image= plot_heatmap(image[0],inspection_code)
    
    # cv2.imwrite("thisimage.png",image)
    #convert to encoding before sending it 
    # prediction_image=img_to_base64_str(prediction_image)
    
    
    os.remove(file_path)
    return {name:prediction_image,confidence:prediction} 

if __name__ == "__main__":


    filename='Final.DCM'
    image=preprocess_image(filename)
    print(image.shape)
    prediction=model.predict(image)
    prediction_image=plot_heatmap(image[0],'thirty')
    
   
    format, imgstr=prediction_image.split(';base64,')
    input_file=os.path.basename(filename)
    output_file_name=re.sub(".png","_pred.json",input_file)
    # print(prediction)
    # print(f"{output_file_name}: {imgstr}")

    





