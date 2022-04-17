import os 



import base64
import tempfile
import numpy as np
import tensorflow as tf
import pydicom
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import Model
from io import BytesIO



model=tf.keras.models.load_model('model.hdf5')


def img_to_base64_str(img):
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    buffered.seek(0)
    img_bytes=buffered.getvalue()
    img_str="data:image/png;base64,"+base64.b64encode(img_bytes).decode()
    
    return img_str

def preprocess(image):
    print("Begining image preprocessing")
    test_image=[]
    print("Reading DICOM image")
    image=pydicom.dcmread(image)
    image=image.pixel_array
    cv2.imwrite('test.png',image)
    image=cv2.imread('test.png')
    image=Image.fromarray(image,'RGB')
    image=image.resize((256,256))
    test_image.append(np.array(image))  
    test_image=np.array(test_image)
    test_image=test_image/255
    print("Done image prep ")
    return test_image
# prediction=model.predict(preprocess(image))

def prediction_image(img,prediction):
    # prediction_classes=tf.keras.applications.densenet.decode_predictions(prediction,top=2)
    # pred_class = np.argmax(prediction)
#    text=None
#    prediction_classes=None


   if prediction[0][0]>0.5:
        text='Pneumothorax Detected with confidence probability of'
        print("Entered image preprocessing pipeline")
        pred_class=np.argmax(prediction)
        last_layer_weights=model.layers[-1].get_weights()[0]
        last_layer_weights_for_pred= last_layer_weights[:,pred_class]
        
        last_conv_model = Model(model.input, model.get_layer("conv5_block16_concat").output)
        last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
        last_conv_output = np.squeeze(last_conv_output)
  
    
        #Upsample/resize the last conv. output to same size as original image
        h = int(img.shape[0]/last_conv_output.shape[0])
        w = int(img.shape[1]/last_conv_output.shape[1])
        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
        
        
        
        heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 1024)), 
                    last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
        
        print("Generating heatmap")
        heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0
            
        plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
        plt.imshow(heat_map, cmap='jet', alpha=0.30)
        
        plt.savefig('heatmap.png')
        
        image=cv2.imread('heatmap.png')
        image=Image.fromarray(image,'RGB')
        image=image.resize((256,256))
        print("Calculating prediction")
        
        
        prediction_classes=prediction[0][0]
        
        return [{'text':text,'probability':prediction_classes*100,'image':img_to_base64_str(image)}]
   else:
       text='No pneumothorax detected with confidence probability of '
       image=Image.fromarray(img,'RGB')
       image=image.resize((256,256))
        
       prediction_classes=prediction[0][1]
       return [{'text':text,'probability':prediction_classes*100,'image':img_to_base64_str(image)}]
    #    return 'No pneumothorax detected with confidence probability of ',prediction[0][1]*100
    
    
    # return prediction

def predict_base64_image(name, contents):
    print("receiving image")
    fd,file_path=tempfile.mkstemp()
    with open(fd,'wb') as f:
        f.write(base64.b64decode(contents))
    print("Stored dicom file")
    image=preprocess_image(file_path)
    
    classes=prediction_image(image[0],model.predict(preprocess(file_path)))
    os.remove(file_path)
    return {name: classes}

if __name__ == '__main__':
    
    file_path='Final.DCM'
    image=preprocess(file_path)
    classes=prediction_image(image[0],model.predict(preprocess(file_path)))
    print(classes)
    
    
