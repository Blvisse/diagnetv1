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
import json
import httplib2
import boto3
import contextlib
import sys
from pydicom.uid import generate_uid
import pymysql
import os
import sys
# import mysql.connector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from sklearn.neighbors import KernelDensity
import joblib
kde=joblib.load('kerneldens.pkl')
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
    
    
host='diagnosoft-rds-ris.c1pejgtriylv.us-east-2.rds.amazonaws.com'

user='admin'
password='Diagnosoft254!'

# @contextlib.contextmanager
def connection_sql():

    try:
        print("Accesing RDS Database")
        connection=pymysql.connect(host=host,
                                
                                user=user,
                                password=password,
                                database='diagnosoft_rds_ris',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
        

        connection=pymysql.connect(host=host,user=user,password=password)

        cursor=connection.cursor()
        print("Connection Succefully established")
        return connection,cursor

    except (Exception, pymysql.Error) as error:
        print("Error while connecting to MySQL", error)
        
# with connection_sql() as connection,cursor:
#     print ("Connectied to rds prepping for data entry")

connection,cursor=connection_sql()
#change here
model=tf.keras.models.load_model('model.hdf5')
anomaly_detector=tf.keras.models.load_model('Encoder.hdf5')
# anomaly_weightts=tf.keras.models.load_model('../anomaly11.hdf5')
URL='https://diagnosoftdicom.azurewebsites.net/pacs/instances'
def IsJson(content):
    try:
        if (sys.version_info >= (3, 0)):
            json.loads(content.decode())
            return True
        else:
            json.loads(content)
            return True
    except Exception as e:
        print(e)
        return False

def img_to_base64_str(img):
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    buffered.seek(0)
    img_bytes=buffered.getvalue()
    img_str="data:image/png;base64,"+base64.b64encode(img_bytes).decode()
    
    return img_str
def preprocess(file_path,p_name,inspection_code):
    print("Begining image preprocessing")
    test_image=[]
    print("Reading DICOM image")
    image=pydicom.dcmread(file_path)
    # image.StudyInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.315222469415623828162094912079'
    # image.SeriesInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.309111601391566216060061725328'
    # image.SOPInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.170420932819315718805816034120'
    image.StudyInstanceUID=generate_uid()
    image.SeriesInstanceUID=generate_uid()
    image.SOPInstanceUID=generate_uid()
    image.PatientName=p_name
    image.PatientID=inspection_code
    pydicom.dcmwrite(file_path,image)
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
    image=pydicom.dcmread(image)
    # image.PhotometricInterpretation = 'RGB'
    # image.SamplesPerPixel = 3
    # image.BitsAllocated = 8
    # image.BitsStored = 8
    # image.HighBit = 7
    # image.add_new(0x00280006, 'US', 0)
    image=image.pixel_array
    # cv2.imwrite('test.png',image)
    # image=cv2.imread('test.png')

    image=Image.fromarray(image)
    image=image.resize((256,256))
    test_image.append(np.array(image))
    test_image=np.array(test_image)
    test_image=test_image/255
    
    return test_image
def upload_image(path,inspection_code):
    
    f=open(path,'rb')
    content=f.read()
    f.close()
    
    
    
    print("Importing file: ")
    
    if IsJson(content):
        print("Ignored JSON file")
        json_count+=1
        return
    try:
        h=httplib2.Http()
        headers={'content-type':'application/dicom'}
        username='orthanc'
        password='orthanc'
        creds_str=username+':'+password
        creds_str_bytes=creds_str.encode('ascii')
        creds_str_bytes_b64=b'Basic '+base64.b64encode(creds_str_bytes)
        headers['authorization']=creds_str_bytes_b64.decode('ascii')
        resp,content=h.request(URL,'POST',body=content,headers=headers)
        
        
        if resp.status == 200:
            
            query='''
            UPDATE diagnosoft_rds_ris.ExaminationRequest
            SET uploaded=1
            WHERE Inspection_code = %s
            
            '''
            cursor.execute(query,(inspection_code,))
            connection.commit()
            print("Updated database record")
            print("Successfully uploaded file: ")
            
        else:
            print("Error uploading file: ")
            print("Is it a dicom file?")
            
    except Exception as e:
        type,value,traceback=sys.exc_info()
        print(str(value))
        print("Unable to connect to Orthanc Server")
        print(e)
    
def prediction_image(img,prediction,inspection_code):
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
        s3.upload_file(Filename="heatmap.png",Bucket="chest-predictions",Key=inspection_code+".png")
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

def check_anomaly(image):
    print("Analyzing image for anomalies")
    density_threshold=167
    reconstruction_error_threshold=0.014
    
    
   
    # print(image.shape)
    print("reshaping image")
    image=image[:,:,:,np.newaxis]
    # print(image.shape)
    
   
    encoded_img=anomaly_detector.predict([[image]])
    encoded_img = [np.reshape(image, (8*8*4)) for image in encoded_img]
    


    #Fit KDE to the image latent data
    
    print("Fitting KDE to image latent data")
    density = kde.score_samples(encoded_img)[0]
    model = Sequential()
    model.add(Conv2D(64, (3, 3),padding='same', input_shape=(256, 256,1)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3),  padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(4, (3, 3),padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D((2, 2), padding='same'))

    #Decoder
    model.add(Conv2D(4, (3, 3),  padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3),padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3),  padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    # model.summary()
    model.load_weights('anomaly11.hdf5')
    print("Reconstructing image")
    reconstruction = model.predict([[image]])
    reconstruction_error = model.evaluate([reconstruction],[[image]], batch_size = 1)[0]
    print("Done")
    print("Final analysis")

    if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
        text="The image is an anomaly"
        return [{'text':text,'with given density':density,'Reconstruction error':reconstruction_error}]
        
    else:
        text="The image is NOT an anomaly"
        return [{'text':text,'with given density':density,'Reconstruction error':reconstruction_error}]

def predict_base64_image(name,patient_name,inspection_code,contents):
    print("receiving image")
    fd,file_path=tempfile.mkstemp()
    with open(fd,'wb') as f:
        f.write(base64.b64decode(contents))
    print("Stored dicom file")
    anomaly_image=preprocess_image(file_path)
    image=preprocess(file_path,patient_name,inspection_code)
    
    upload_image(file_path,int(inspection_code))
    anomaly_analysis=check_anomaly(anomaly_image)
    classes=prediction_image(image[0],model.predict(image),inspection_code)
    
    os.remove(file_path)
    return {name: classes,patient_name:anomaly_analysis}

if __name__ == '__main__':
    
    file_path='1-03.DCM'
    
    # image=preprocess(file_path,'Brain_Scan_2','999')
    image=preprocess_image(file_path)
    # upload_image(file_path,999)
    print(check_anomaly(image))
    # classes=prediction_image(image[0],model.predict(preprocess(file_path,'blaise_final_test')),'34567')
    # print(classes)
    
    
