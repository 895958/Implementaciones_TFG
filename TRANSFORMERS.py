

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

from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


#Sequence model
def get_sequence_model():
    
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

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
    
    # seq_model.save(classifierFilePath+'modelo_TRANSFORMERS_user_'+str(sujeto)+'.h5')
    
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
    
    with open('results_TRANSFORMERS_all.pkl','wb') as fid:
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