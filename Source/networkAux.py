import numpy as np
import cv2 as cv
import tensorflow.keras as K
import os
import random
import csv
import cAugmentation

# ====================================================================================================================
def GranulometryNet(nOutNodes, input_depth=3, regularisation = None, doBatchnorm = False, doMultiScale = False):
    inputs = K.Input(shape=(None, None, 3)) # static batch-size to enable target loss computation
    
    # ENCODER
    x = K.layers.Conv2D(32, 3, padding = "same", kernel_regularizer=regularisation)(inputs)
    if doBatchnorm:
        x = K.layers.BatchNormalization()(x)
    x1d = K.layers.Activation("relu")(x)
    
    if doMultiScale:
        x2d = R_S_Module_down_multiscale(nFilter=32, doBatchnorm=doBatchnorm, regularisation=regularisation)(x1d)
        x3d = R_S_Module_down_multiscale(nFilter=64, doBatchnorm=doBatchnorm, regularisation=regularisation)(x2d)
        x4d = R_S_Module_down_multiscale(nFilter=128, doBatchnorm=doBatchnorm, regularisation=regularisation)(x3d)
        x5d = R_S_Module_down_multiscale(nFilter=256, doBatchnorm=doBatchnorm,regularisation=regularisation)(x4d)
    else:
        x2d = R_S_Module_down(nFilter=32, doBatchnorm=doBatchnorm, regularisation=regularisation)(x1d)
        x3d = R_S_Module_down(nFilter=64, doBatchnorm=doBatchnorm, regularisation=regularisation)(x2d)
        x4d = R_S_Module_down(nFilter=128, doBatchnorm=doBatchnorm, regularisation=regularisation)(x3d)
        x5d = R_S_Module_down(nFilter=256, doBatchnorm=doBatchnorm,regularisation=regularisation)(x4d)
    
    x = K.layers.Conv2D(filters=nOutNodes, kernel_size=1, padding="same", kernel_regularizer=regularisation)(x5d)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Softmax()(x)
    
    model = K.Model(inputs=inputs, outputs=x)
    return model 


# ====================================================================================================================
class R_S_Module_down(K.layers.Layer):
    def __init__(self, nFilter, doBatchnorm=False, regularisation=None, name=None):
        super(R_S_Module_down, self).__init__(name=name)
        self.nFilter=nFilter
        self.doBatchnorm=doBatchnorm
        self.regularisation=regularisation
        self.bn = K.layers.BatchNormalization()
        self.conv = K.layers.Conv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.conv_stride2 = K.layers.Conv2D(filters=nFilter, kernel_size=3, strides=2, padding="same", kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.sepconv = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.maxpool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")
        self.relu = K.layers.Activation("relu")
        self.add = K.layers.Add()
        
    # ____________________________________________
    def call(self, input_tensor):
        residual = self.conv_stride2(input_tensor)
        if self.doBatchnorm:
            residual = self.bn(residual)
        
        x = self.conv(input_tensor)
        if self.doBatchnorm:
            x = self.bn(x)
        x=self.relu(x)
        
        x = self.sepconv(x)
        if self.doBatchnorm:
            x = self.bn(x)
        x = self.maxpool(x)
        
        x=self.add([x, residual])
        x=self.relu(x)
        
        return x
    
    
# ====================================================================================================================
class R_S_Module_down_multiscale(K.layers.Layer):
    def __init__(self, nFilter, doBatchnorm = False, regularisation=None, name=None):
        super(R_S_Module_down_multiscale, self).__init__(name=name)
        self.doBatchnorm=doBatchnorm
        self.conv_stride2 = K.layers.Conv2D(filters=nFilter, kernel_size=3, strides=2, padding="same", kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.convDilate1 = K.layers.Conv2D(filters=int(nFilter/2), kernel_size=3, padding="same", dilation_rate = (1,1), kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.convDilate2 = K.layers.Conv2D(filters=int(nFilter/4), kernel_size=3, padding="same", dilation_rate = (2,2), kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.convDilate4 = K.layers.Conv2D(filters=int(nFilter/4), kernel_size=3, padding="same", dilation_rate = (4,4), kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.sepconv = K.layers.SeparableConv2D(filters=nFilter, kernel_size=3, padding="same", kernel_regularizer=regularisation, kernel_initializer='he_normal')
        self.maxpool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")
        self.bn = K.layers.BatchNormalization()
        self.relu = K.layers.Activation("relu")
        self.concat = K.layers.Concatenate()
        self.add = K.layers.Add()
    
    # ____________________________________________
    def call(self, input_tensor):
        residual = self.conv_stride2(input_tensor)
        if self.doBatchnorm:
            residual = self.bn(residual)
        
        x1 = self.convDilate1(input_tensor)
        x2 = self.convDilate2(input_tensor)
        x3 = self.convDilate4(input_tensor)
        x = self.concat([x1,x2,x3])
        if self.doBatchnorm:
            x = self.bn(x)
        x = self.relu(x)
        
        x = self.sepconv(x)
        if self.doBatchnorm:
            x = self.bn(x)
        x = self.maxpool(x)
        
        x = self.add([x, residual])
        
        x = self.relu(x)
        
        return x
    

# ====================================================================================================================================================
def generator_classification(inputPathImg, inputPathLabel, batchsize, nOutNodes, in_size=(550,750), Augmenter=cAugmentation.cAugmentation()):
    # in_size in order (columns, rows)
    # get list of all image names inside the path
    imgList = os.listdir(inputPathImg)
    random.shuffle(imgList)    
    
    # infinit loop    
    i = 0
    while True:
        
        
        images = np.zeros(shape=(batchsize, in_size[1], in_size[0], 3), dtype=np.float)
        labels = np.zeros(shape=(batchsize, nOutNodes), dtype=np.float)
        for b in range(batchsize):
            if (i==len(imgList)):
                i = 0
                random.shuffle(imgList)
                
            # Read input image
            fnCurrImg = os.path.join(inputPathImg, imgList[i])
            filename, file_extension = os.path.splitext(imgList[i])
            if not fnCurrImg.endswith(".jpg") and not fnCurrImg.endswith(".png") and not fnCurrImg.endswith(".JPG"):
                continue
           
            x = cv.imread(fnCurrImg, cv.IMREAD_COLOR).astype('float32')/255.                        
            
            x = Augmenter.augment_data(x)
            
            # Read Label 
            fnCurrLabel = os.path.join(inputPathLabel, '%s.txt' %(filename))
            f = open(fnCurrLabel, "r")
            f = f.read()            
            y = int(f[0])
            
            # Update output
            images[b,:,:,:] = x
            labels[b,y]=1
            
            i+=1
        yield (images,labels)
        
# ====================================================================================================================================================
def generator_regression(inputPathImg, inputPathLabel, batchsize, nOutNodes, in_size=(550,750), Augmenter=cAugmentation.cAugmentation()):
    # in_size in order (columns, rows)
    # get list of all image names inside the path
    imgList = os.listdir(inputPathImg)
    random.shuffle(imgList)    
    
    # infinit loop
    i = 0
    
    while True:        
        
        images = np.zeros(shape=(batchsize, in_size[1], in_size[0], 3), dtype=np.float)
        labels = np.zeros(shape=(batchsize, nOutNodes), dtype=np.float)
        for b in range(batchsize):
            if (i==len(imgList)):
                i = 0
                random.shuffle(imgList)
                
            # Read input image
            fnCurrImg = os.path.join(inputPathImg, imgList[i])
            filename, file_extension = os.path.splitext(imgList[i])
            if not fnCurrImg.endswith(".jpg") and not fnCurrImg.endswith(".png") and not fnCurrImg.endswith(".JPG"):
                continue
           
            x = cv.imread(fnCurrImg, cv.IMREAD_COLOR).astype('float32')/255.         
            
            x = Augmenter.augment_data(x)
            
            # Read Label 
            fnCurrLabel = os.path.join(inputPathLabel, '%s.txt' %(filename))
            reader = csv.reader(open(fnCurrLabel), delimiter=" ")
            
            y = np.zeros((nOutNodes), np.float)
            count=0
            for row in reader:
                y[count] = float(row[0])
                count+=1
            
            # Update output
            images[b,:,:,:] = x
            labels[b,:]=y
            
            i+=1
        yield (images,labels)        


# ====================================================================================================================
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=15, min_lr=1e-09, verbose=1)

# ====================================================================================================================
checkpointer = K.callbacks.ModelCheckpoint('DeepGranNet_{epoch:03d}-{loss:.4f}.hdf5',
                                               verbose=0, 
                                               monitor='loss',
                                               save_best_only=True, 
                                               save_weights_only=True, 
                                               mode='auto',
                                               period=1)