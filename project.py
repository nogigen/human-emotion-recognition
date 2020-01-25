
# coding: utf-8

# In[ ]:


# pip install keras


# In[2]:


# load the data
import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt;
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
# size of the pic
pic_size = 48

# base path
base_path = "./images/"

plt.figure(0, figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path + "train/"):
    for i in range(5):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = load_img(base_path + "train/" + expression + "/" + os.listdir(base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[3]:


# check the number of pictures
for expression in os.listdir(base_path + "train"):
    print(str(len(os.listdir(base_path + "train/" + expression))) + " " + expression + " images")


# In[4]:


# split data.
# 4 convolutional layers
# 2 fully connected layers
from keras.preprocessing.image import ImageDataGenerator
# number of images to feed into the NN for every batch
batch_size = 128

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(base_path + "train",
                                                    target_size=(pic_size,pic_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(base_path + "validation",
                                                    target_size=(pic_size,pic_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)


# In[5]:


# setup CNN
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# number of possible label values
nb_classes = 7

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[6]:


get_ipython().run_cell_magic('time', '', '# train the model\n\n# number of epochs to train the NN\nepochs = 10\n\nfrom keras.callbacks import ModelCheckpoint\n\ncheckpoint = ModelCheckpoint("model_weights.h5", monitor=\'val_acc\', verbose=1, save_best_only=True, mode=\'max\')\ncallbacks_list = [checkpoint]\n\nhistory = model.fit_generator(generator=train_generator,\n                                steps_per_epoch=train_generator.n//train_generator.batch_size,\n                                epochs=epochs,\n                                validation_data = validation_generator,\n                                validation_steps = validation_generator.n//validation_generator.batch_size,\n                                callbacks=callbacks_list\n                                )')


# In[7]:


# serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[115]:


# plot the evolution of Loss on the train and validation sets

import matplotlib.pyplot as plt

xlabel = [""+str(2 * x + 1) for x in range(10)]
plt.figure(figsize=(10,10))
plt.xticks(np.arange(0, epochs + 2, step=2) , tuple(xlabel))
plt.title('Loss - Iteration Number (Optimizer = Adam)')
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Iteration Number', fontsize = 16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')


# In[116]:


# plot the evolution of Accuracy on the train and validation sets
xlabel = [""+str(2 * x + 1) for x in range(10)]
plt.figure(figsize=(10,10))
plt.xticks(np.arange(0, epochs + 2, step=2) , tuple(xlabel))
print(np.arange(2, epochs + 2, step=2))
plt.title('Loss - Iteration Number (Optimizer = Adam)')
plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Iteration Number', fontsize = 16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
print(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[33]:


# show the confusion matrix of our predictions

# compute predictions
predictions = model.predict_generator(generator=validation_generator)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()


# In[66]:


# testing.
filenames=validation_generator.filenames
classes = ["angry", "disgust", "fear","happy","neutral","sad","surprise"]
print("file name is :" , filenames[0])
print("number of files are :", str(len(filenames)))
print("classes are :", str(class_names))
print("first prediction is :", str(predictions[0]))
print("length of y_test :", str(len(y_test)))
print("length of y_pred :", str(len(y_pred)))
print("first test class is :", str(y_test[0]))
print("first pred class is :", str(y_pred[0]))


# In[76]:


# get 5 true predictions of "angry" picture
# get 5 false predictions of "angry" picture
true_angry_path = []
false_angry_path = []
false_angry_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 0 and y_test[iterate] == y_pred[iterate] and len(true_angry_path) != 5:
        true_angry_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 0 and y_test[iterate] != y_pred[iterate] and len(false_angry_path) != 5:
            false_angry_path.append(filenames[iterate])
            false_angry_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_angry_path)
print(false_angry_path)
print(false_angry_class)


# In[80]:


# plot 5 true predictions of "angry" picture
# plot 5 false predictions of "angry" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "anger")
        if i == 0:
            img = load_img(base_path + "validation/" +true_angry_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "anger")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_angry_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_angry_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[88]:


# get 5 true predictions of "disgust" picture
# get 5 false predictions of "disgust" picture
true_disgust_path = []
false_disgust_path = []
false_disgust_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 1 and y_test[iterate] == y_pred[iterate] and len(true_disgust_path) != 5:
        true_disgust_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 1 and y_test[iterate] != y_pred[iterate] and len(false_disgust_path) != 5:
            false_disgust_path.append(filenames[iterate])
            false_disgust_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_disgust_path)
print(false_disgust_path)
print(false_disgust_class)


# In[89]:


# plot 5 true predictions of "disgust" picture
# plot 5 false predictions of "disgust" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "disgust")
        if i == 0:
            img = load_img(base_path + "validation/" +true_disgust_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "disgust")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_disgust_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_disgust_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[91]:


# get 5 true predictions of "fear" picture
# get 5 false predictions of "fear" picture
true_fear_path = []
false_fear_path = []
false_fear_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 2 and y_test[iterate] == y_pred[iterate] and len(true_fear_path) != 5:
        true_fear_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 2 and y_test[iterate] != y_pred[iterate] and len(false_fear_path) != 5:
            false_fear_path.append(filenames[iterate])
            false_fear_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_fear_path)
print(false_fear_path)
print(false_fear_class)


# In[92]:


# plot 5 true predictions of "fear" picture
# plot 5 false predictions of "fear" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "fear")
        if i == 0:
            img = load_img(base_path + "validation/" +true_fear_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "fear")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_fear_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_fear_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[94]:


# get 5 true predictions of "happy" picture
# get 5 false predictions of "happy" picture
true_happy_path = []
false_happy_path = []
false_happy_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 3 and y_test[iterate] == y_pred[iterate] and len(true_happy_path) != 5:
        true_happy_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 3 and y_test[iterate] != y_pred[iterate] and len(false_happy_path) != 5:
            false_happy_path.append(filenames[iterate])
            false_happy_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_happy_path)
print(false_happy_path)
print(false_happy_class)


# In[95]:


# plot 5 true predictions of "happy" picture
# plot 5 false predictions of "happy" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "happy")
        if i == 0:
            img = load_img(base_path + "validation/" +true_happy_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "happy")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_happy_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_happy_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[97]:


# get 5 true predictions of "neutral" picture
# get 5 false predictions of "neutral" picture
true_neutral_path = []
false_neutral_path = []
false_neutral_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 4 and y_test[iterate] == y_pred[iterate] and len(true_neutral_path) != 5:
        true_neutral_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 4 and y_test[iterate] != y_pred[iterate] and len(false_neutral_path) != 5:
            false_neutral_path.append(filenames[iterate])
            false_neutral_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_neutral_path)
print(false_neutral_path)
print(false_neutral_class)


# In[98]:


# plot 5 true predictions of "neutral" picture
# plot 5 false predictions of "neutral" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "neutral")
        if i == 0:
            img = load_img(base_path + "validation/" +true_neutral_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "neutral")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_neutral_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_neutral_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[100]:


# get 5 true predictions of "sad" picture
# get 5 false predictions of "sad" picture
true_sad_path = []
false_sad_path = []
false_sad_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 5 and y_test[iterate] == y_pred[iterate] and len(true_sad_path) != 5:
        true_sad_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 5 and y_test[iterate] != y_pred[iterate] and len(false_sad_path) != 5:
            false_sad_path.append(filenames[iterate])
            false_sad_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_sad_path)
print(false_sad_path)
print(false_sad_class)


# In[101]:


# plot 5 true predictions of "sad" picture
# plot 5 false predictions of "sad" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "sad")
        if i == 0:
            img = load_img(base_path + "validation/" +true_sad_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "sad")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_sad_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_sad_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 


# In[103]:


# get 5 true predictions of "surprise" picture
# get 5 false predictions of "surprise" picture
true_surprise_path = []
false_surprise_path = []
false_surprise_class = []
count = 0
iterate = 0
while count != 10:
    if y_test[iterate] == 6 and y_test[iterate] == y_pred[iterate] and len(true_surprise_path) != 5:
        true_surprise_path.append(filenames[iterate])
        count += 1
    else:
        if y_test[iterate] == 6 and y_test[iterate] != y_pred[iterate] and len(false_surprise_path) != 5:
            false_surprise_path.append(filenames[iterate])
            false_surprise_class.append(y_pred[iterate])
            count += 1
    iterate += 1

print(true_surprise_path)
print(false_surprise_path)
print(false_surprise_class)


# In[104]:


# plot 5 true predictions of "surprise" picture
# plot 5 false predictions of "surprise" picture

plt.figure(0, figsize=(15,5))
cpt = 0
true_count = 0
false_count = 0
for i in range(2):
    for j in range(5):
        cpt = cpt + 1
        plt.subplot(2,5,cpt)
        plt.ylabel("True label: " + "surprise")
        if i == 0:
            img = load_img(base_path + "validation/" +true_surprise_path[true_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label: " + "surprise")
            true_count += 1
        else:
            img = load_img(base_path + "validation/" +false_surprise_path[false_count].replace("\\","/"), target_size=(pic_size, pic_size))
            plt.xlabel("Pred label:" + classes[false_surprise_class[false_count]])
            false_count += 1           

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show() 

