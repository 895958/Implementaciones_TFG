
from tensorflow import keras
# from imutils import paths
#import tensorflow.keras.backend as K
from keras import backend as K
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import pickle


# =============================================================================
# PARAMS
# =============================================================================

IMG_SIZE = 128#224
EPOCHS = 60

MAX_SEQ_LENGTH = 50#60
NUM_FEATURES = 1024#2048

base_dir = "../"

PREPARE = False

classifierFilePath= "./videoClassifier/"

# =============================================================================
# FUNCTIONS
# =============================================================================

def crop_center_square(frame): #Se ajusta el tamaño de la imagenes de forma que estas sean cuadradas
    y,x = frame.shape[0:2]
    min_dim = min(y,x)
    start_x = (x//2) - (min_dim//2)
    start_y = (y//2) - (min_dim//2)
    return frame[start_y : start_y+min_dim, start_x : start_x + min_dim]

def load_video(path,max_frame=0, resize=(IMG_SIZE,IMG_SIZE)):
    
    frames = []
    paths = glob(os.path.join(path,'*.jpg'))

    for img in paths:
        frame = cv2.imread(img, cv2.IMREAD_COLOR)
        frame = crop_center_square(frame)
        frame = cv2.resize(frame,resize)
        frame = frame[:,:,[2,1,0]] #El formato de open CV es BGR lo pasamos a RGB
        frames.append(frame)
   
        if len(frames) == max_frame:
            break
       
        # print(np.shape(frames))
        # finally:
            #cap.release()
            # plt.imshow(frames[0])#PRUEBA
    return np.array(frames) #Tramas pasadas al formato de np array
 
def build_feature_extractor():
    
    # feature_extractor = keras.applications.InceptionV3( #Utilizamos una red convolucional ya creada
    #                                                     weights = "imagenet",
    #                                                     include_top = False,#Solo utilizamos la base
    #                                                     pooling = "avg",
    #                                                     input_shape = (IMG_SIZE,IMG_SIZE,3),
    #                                                     )
    
    # preprocess_input = keras.applications.inception_v3.preprocess_input #Prepara los datos para el modelo de forma que se convierten en pixels entre 1 y -1

    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE,IMG_SIZE,3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    
    return keras.Model(inputs, outputs, name = "feature_extractor")

def prepare_all_videos(df, label_processor, feature_extractor, root_dir):

    video_paths = df["Ruta"].values.tolist()
    sujetos  = df["Sujeto"].values.tolist()
    labels = df["Accion"].values
    labels = label_processor(labels[...,None]).numpy()
    #Frame mask necesarias para la posterior impplementacin de  GRU

    frameInfo = {}

    #For each video
    for idx, path in enumerate(video_paths):
        
        print(idx)
        frameInfo[path]= {'frameFeatures':[],'frames':[]}
        
        #Gather all its frames and add a batch dimension
        frames = load_video(os.path.join(root_dir, path))
        # frames = frames[None,...]
        video_length = frames.shape[1]
        length = min(MAX_SEQ_LENGTH,video_length)
        temp_frame_features = feature_extractor.predict(frames,verbose=0)
        temp_frame_features = temp_frame_features[:length,:]
        print(np.shape(temp_frame_features))
        frameInfo[path]['frameFeatures']=temp_frame_features
        frameInfo[path]['frames']=frames
        frameInfo[path]['label']=labels[idx]
        frameInfo[path]['subject']=sujetos[idx]
    
    return frameInfo,labels,sujetos


# This is a sample of a scheduler I used in the past
def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

#Sequence model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()
    print(class_vocab)
    
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    
    x = keras.layers.LSTM(80,return_sequences=True)(frame_features_input)
    x = keras.layers.LSTM(80)(x)
    x = keras.layers.Dense(200, activation = "relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    output =  keras.layers.Dense(len(class_vocab),activation="softmax")(x)
    
    rnn_model = keras.Model(frame_features_input, output)
    
    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
    )
    
    return rnn_model

# =============================================================================
# MAIN
# =============================================================================

df = pd.read_csv('../RGB1_split_v1.csv',delimiter=';')

feature_extractor = build_feature_extractor()

activity_map = {'AC10': 'Talking to passenger', #Mapeamos cada numero con su significado
                'AC1': 'Safe driving', 
                'AC2': 'Doing hair and makeup', 
                'AC3': 'Adjusting radio', 
                'AC4': 'GPS operating', 
                'AC5': 'Writing message using right hand', 
                'AC6': 'Writing message using left hand', 
                'AC7': 'Talking phone using right hand', 
                'AC8': 'Talking phone using left hand', 
                'AC9': 'Having picture'}
                # 'AC11': 'Singing or dancing',
                # 'AC12': 'Fatigue and somnolence',
                # 'AC13': 'Drinking using right hand',
                # 'AC14': 'Drinking using left hand',
                # 'AC15': 'Reaching behind',
                # 'AC16': 'Smoking'}

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["Accion"]))
print(label_processor.get_vocabulary())

if PREPARE == True:

    data, labels, sujetos = prepare_all_videos(df,label_processor,feature_extractor,base_dir)
    
    with open('all_data.pkl','wb') as fid:
        pickle.dump(data,fid)
    with open('all_labels.pkl','wb') as fid:
        pickle.dump(labels,fid)
    with open('all_subjects.pkl','wb') as fid:
        pickle.dump(sujetos,fid)
        
else:
    
    with open('all_data.pkl','rb') as fid:
        data = pickle.load(fid)
    with open('all_labels.pkl','rb') as fid:
        labels = pickle.load(fid)
    with open('all_subjects.pkl','rb') as fid:
        sujetos = pickle.load(fid)
 
# LOOP

results = {}

for sujeto in np.unique(sujetos):
    
    results[sujeto] = {}
    
    X_train = np.array([data[key]['frameFeatures'] for key in data.keys() if data[key]['subject']!=sujeto])
    y_train = np.squeeze([data[key]['label'] for key in data.keys() if data[key]['subject']!=sujeto])
    
    X_test = np.array([data[key]['frameFeatures'] for key in data.keys() if data[key]['subject']==sujeto])
    y_test = np.squeeze([data[key]['label'] for key in data.keys() if data[key]['subject']==sujeto])

    keras.backend.clear_session()

    seq_model = get_sequence_model()

    checkpoint = keras.callbacks.ModelCheckpoint(
          classifierFilePath, save_weights_only = True, save_best_only = True, verbose=1
      )    

    history = seq_model.fit(
          X_train,
          y_train,
          validation_split = 0.1,
          epochs = EPOCHS,
          shuffle=False,
          callbacks = [checkpoint])
    
    # seq_model.save(classifierFilePath+'modelo_CCN+RRN_user_'+str(sujeto)+'.h5')

    # =============================================================================
    # TEST Leave one out
    # =============================================================================

    y_pred = seq_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred,1)
    
    results[sujeto]['ytest'] = y_test
    results[sujeto]['ypred'] = y_pred_classes
    results[sujeto]['ypred_prob'] = y_pred
    results[sujeto]['accuracy'] = sum(y_pred_classes==y_test)/len(y_test)
    results[sujeto]['history'] = history
    
    print('Accuracy en sujeto de test',sujeto,':',str(results[sujeto]['accuracy']))
    
    with open('results_cnnrnn_all.pkl','wb') as fid:
        pickle.dump(results,fid)


training_accs = [results[user]['history'].history['accuracy'] for user in results.keys()]
val_accs = [results[user]['history'].history['val_accuracy'] for user in results.keys()]

plt.plot(np.mean(training_accs,0),'b')
plt.fill_between(range(60),np.mean(training_accs,0)-np.std(training_accs,0),np.mean(training_accs,0)+np.std(training_accs,0),color='b',alpha=0.3,label='Training accuracy')

plt.plot(np.mean(val_accs,0),'red')
plt.fill_between(range(60),np.mean(val_accs,0)-np.std(val_accs,0),np.mean(val_accs,0)+np.std(val_accs,0),color='r',alpha=0.3,label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()