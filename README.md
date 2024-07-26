# 1.-Feature-Extraction-for-Signature-Verification-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, 
ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array
import gc
from tensorflow.keras.models import Model
from PIL import Image
import pickle
train_dir="../input/signature-verification-dataset-iraninan/train"
test_dir="../input/signature-verification-dataset-iraninan/test"
img = plt.imread('../input/signature-verification-dataset-iraninan/train/04/04-01.jpg')
plt.imshow(img)
img = plt.imread('../input/signature-verification-dataset-iraninan/train/04-f/04-f-01.jpg')
plt.imshow(img)
train_data_names = []
test_data_names = []
train_data = []
train_labels = []
for per in os.listdir('../input/signature-verification-dataset-iraninan/train'):
 for data in glob.glob('../input/signature-verification-dataset-iraninan/train/'+per+'/*.*'):
 
 train_data_names.append(data)
 img = cv2.imread(data)
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 img = cv2.resize(img, (SIZE,SIZE))
 train_data.append([img])
 if per[-1]=='f':
 train_labels.append(np.array(1))
 else:
 train_labels.append(np.array(0))
train_data = np.array(train_data)/255.0
train_labels = np.array(train_labels)
#Test Data
test_data = []
test_labels = []
for per in os.listdir('../input/signature-verification-dataset-iraninan/test'):
 for data in glob.glob('../input/signature-verification-dataset-iraninan/test/'+per+'/*.*'):
 test_data_names.append(data)
 img = cv2.imread(data)
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 img = cv2.resize(img, (SIZE,SIZE))
 test_data.append([img])
 if per[-1]=='f':
 test_labels.append(np.array(1))
 else:
 test_labels.append(np.array(0))
test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)
train_labels[1:1000]
with open('./train_data_names.pkl', 'wb') as fp:
 pickle.dump(train_data_names, fp)
with open('./test_data_names.pkl', 'wb') as fp:
 pickle.dump(test_data_names, fp)
# Categorical labels
print(train_labels.shape)
train_labels = to_categorical(train_labels)
print(train_data.shape)
# Reshaping
train_data = train_data.reshape(-1, SIZE,SIZE, 3)
test_data = test_data.reshape(-1, SIZE,SIZE, 3)
print(train_data.shape)
print(test_data.shape)
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_)
model = Sequential()
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='sigmoid'))
model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
 metrics=['accuracy'])
model.summary()
earlyStopping = EarlyStopping(monitor='val_loss',
 min_delta=0,
 patience=3,
 verbose=1)
early_stop=[earlyStopping]
progess = model.fit(train_data,train_labels, batch_size=BS,epochs=EPOCHS, 
callbacks=early_stop,validation_split=.3)
acc = progess.history['accuracy']
val_acc = progess.history['val_accuracy']
loss = progess.history['loss']
val_loss = progess.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
intermediate_layer_model = Model(inputs=model.input,
 outputs=model.layers[-2].output)
intermediate_output_train = intermediate_layer_model.predict(train_data)
intermediate_output_test = intermediate_layer_model.predict(test_data)
np.save('./VGG16_Adam_train', intermediate_output_train)
np.save('./VGG16_Adam_test', intermediate_output_test)
model.save('kerasVggSigFeatures.h5')
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_)
model = Sequential()
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='softmax'))
model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
 metrics=['accuracy'])
model.summary()
earlyStopping = EarlyStopping(monitor='val_loss',
 min_delta=0,
 patience=3,
 verbose=1)
early_stop=[earlyStopping]
train_labels = np.argmax(train_labels, axis=1)
progess = model.fit(train_data,train_labels, batch_size=BS,epochs=EPOCHS, 
callbacks=early_stop,validation_split=.3)
acc = progess.history['accuracy']
val_acc = progess.history['val_accuracy']
loss = progess.history['loss']
val_loss = progess.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
intermediate_layer_model = Model(inputs=model.input,
 outputs=model.layers[-2].output)
intermediate_output_train = intermediate_layer_model.predict(train_data)
intermediate_output_test = intermediate_layer_model.predict(test_data)
np.save('./InceptionV3_Adam_train', intermediate_output_train)
np.save('./InceptionV3_Adam_test', intermediate_output_test)
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_)
model = Sequential()
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='softmax'))
model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adagrad(lr=1e-4),
 metrics=['accuracy'])
model.summary()
earlyStopping = EarlyStopping(monitor='val_loss',
 min_delta=0,
 patience=3,
 verbose=1)
early_stop=[earlyStopping]
progess = model.fit(train_data,train_labels, batch_size=BS,epochs=EPOCHS, 
callbacks=early_stop,validation_split=.3)
acc = progess.history['accuracy']
val_acc = progess.history['val_accuracy']
loss = progess.history['loss']
val_loss = progess.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
intermediate_layer_model = Model(inputs=model.input,
 outputs=model.layers[-2].output)
intermediate_output_train = intermediate_layer_model.predict(train_data)
intermediate_output_test = intermediate_layer_model.predict(test_data)
np.save('./InceptionV3_Adagrad_train', intermediate_output_train)
np.save('./InceptionV3_Adagrad_test', intermediate_output_test)
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from scipy.spatial import distance
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
def dataExt(model, optimizer):
 # Load feature ext Data
 filesTrain = "/kaggle/working/train_data_names.pkl"
 filesTest = "/kaggle/working/test_data_names.pkl"
 #../input/feature-extracted/InceptionV3_features/InceptionV3_Adagrad_test.npy
 pathTrain = "/kaggle/working/VGG16_Adam_train.npy"
 pathTest = "/kaggle/working/VGG16_Adam_test.npy"
 
 # unload pickle the file names
 with open(filesTrain,'rb') as f:
 file_train_list = np.load(f, allow_pickle=True)
 
 with open(filesTest,'rb') as f:
 file_test_list = np.load(f, allow_pickle=True)
 # data preprocessing 
 file_train_list = [i[55:] for i in file_train_list]
 file_test_list = [i[54:] for i in file_test_list]
 
 feat_train_np = np.load(pathTrain)
 feat_test_np = np.load(pathTest)
 # return all the data of features of specific model
 return file_train_list, file_test_list, feat_train_np, feat_test_np
def PolySVM(model, optimizer, X_train, y_train, X_test, y_test):
 print("POLY SVM", model, optimizer) 
 
 a,b,c,d = dataExt(model, optimizer)
 
 def name2feat(string):
 try:
 index = a.index(string)
 return c[index]
 except:
 index = b.index(string)
 return d[index] 
 X_train['img1'] = X_train['img1'].apply(name2feat)
 X_train['img2'] = X_train['img2'].apply(name2feat)
 X_test['img1'] = X_test['img1'].apply(name2feat)
 X_test['img2'] = X_test['img2'].apply(name2feat)
 
 new_data_train = []
 for index,row in X_train.iterrows():
 new_list = list(row[0])
 new_list.extend(row[1])
 new_data_train.append(new_list)
 
 new_data_test = []
 for index,row in X_test.iterrows():
 new_list = list(row[0])
 new_list.extend(row[1])
 new_data_test.append(new_list)
 
 Model = svm.SVC(kernel='poly') #rbf by default svm.SVC()
 Model.fit(new_data_train, y_train)
 with open('svm-'+model+'-'+optimizer+'.pkl','wb') as f:
 pickle.dump(Model,f)
 y_pred = Model.predict(new_data_test)
 print("Acuracy", accuracy_score(y_test, y_pred))
 print("P,R,F1:",precision_recall_fscore_support(y_test, y_pred, average='macro'))
 df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred,normalize = 'true'), index = [i for i in "01"],
 columns = [i for i in "01"])
 plt.figure(figsize = (10,7))
 sn.heatmap(df_cm, annot=True)
import os
import csv
def find_files_in_folder(path, folder_name):
 folder_path = None
 for root, dirs, files in os.walk(path):
 if folder_name in dirs:
 folder_path = os.path.join(root, folder_name)
 break
 img1 = []
 if folder_path:
 files = os.listdir(folder_path)
 for file in files:
 folder_name = os.path.dirname(file)
 file_name = os.path.basename(file)
 img1.append(file_name)
 return img1
 else:
 return None
# Example usage
folder_train_path = '../input/signature-verification-dataset-iraninan/train' # Replace with the actual 
folder path
folder_test_path = '../input/signature-verification-dataset-iraninan/test' # Replace with the actual folder 
path
# Get all files in the folder
files_train = os.listdir(folder_train_path)
files_test = os.listdir(folder_test_path)
img1_train = []
img2_train = []
img1_test = []
img2_test = []
for file in files_train:
 files_in_folder = find_files_in_folder(folder_train_path, file)
 if "-f" in file:
 img1_train.append(file)
 else:
 img2_train.append(file)
 
for file in files_test:
 files_in_folder = find_files_in_folder(folder_test_path, file)
 if "-f" in file:
 img1_test.append(file)
 else:
 img2_test.append(file)
# Write folder name and file name to a CSV file
with open('train_data.csv', 'w', newline='') as csvfile:
 writer = csv.writer(csvfile)
 writer.writerow(['img1', 'img2', 'target'])
 for real in sorted(img2_train):
 files_in_real_folder = find_files_in_folder(folder_train_path, real)
 for real_file in files_in_real_folder:
 for forg in sorted(img1_train):
 f = forg.split('-f')
 if f[0] == real:
 files_in_forg_folder = find_files_in_folder(folder_train_path, forg)
 for forg_file in files_in_forg_folder:
 writer.writerow([real+'/'+real_file, forg+'/'+forg_file, 1])
 break
 for real_file_2 in files_in_real_folder:
 if real_file != real_file_2:
 writer.writerow([real+'/'+real_file, real+'/'+real_file_2, 0])
 
# Write folder name and file name to a CSV file
with open('test_data.csv', 'w', newline='') as csvfile:
 writer = csv.writer(csvfile)
 writer.writerow(['img1', 'img2', 'target'])
 for real in sorted(img2_test):
 files_in_real_folder = find_files_in_folder(folder_test_path, real)
 for real_file in files_in_real_folder:
 for forg in sorted(img1_test):
 f = forg.split('-f')
 if f[0] == real:
 files_in_forg_folder = find_files_in_folder(folder_test_path, forg)
 for forg_file in files_in_forg_folder:
 writer.writerow([real+'/'+real_file, forg+'/'+forg_file, 1])
 break
 for real_file_2 in files_in_real_folder:
 if real_file != real_file_2:
 writer.writerow([real+'/'+real_file, real+'/'+real_file_2, 0])
# Write folder name and file name to a CSV file
with open('data.csv', 'w', newline='') as csvfile:
 writer = csv.writer(csvfile)
 writer.writerow(['img1', 'img2', 'target'])
 for real in sorted(img2_train):
 files_in_real_folder = find_files_in_folder(folder_train_path, real)
 for real_file in files_in_real_folder:
 for real2 in sorted(img2_train):
 files_in_real_folder2 = find_files_in_folder(folder_train_path, real2)
 for real_file2 in files_in_real_folder2:
 if real_file != real_file2 and int(real) < 32 and int(real2) < 32:
 if real == real2:
 target = 0
 else:
 target = 1
 writer.writerow([real+"/"+real_file, real2+'/'+real_file2, target])
%%time
data = pd.read_csv('/kaggle/working/train_data.csv')
X_data = data[['img1','img2']]
y_data = data[['target']]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33)
PolySVM('VGG16','RMSprop', X_train, y_train, X_test, y_test)
%%time
data = pd.read_csv('/kaggle/working/train_data.csv')
X_data = data[['img1','img2']]
y_data = data[['target']]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33)
PolySVM('InceptionV3','Adam', X_train, y_train, X_test, y_test)
%%time
data = pd.read_csv('/kaggle/working/train_data.csv')
X_data = data[['img1','img2']]
y_data = data[['target']]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33)
LogReg('InceptionV3','Adagrad', X_train, y_train, X_test, y_test)
