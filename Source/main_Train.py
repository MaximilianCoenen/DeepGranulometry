import tensorflow as tf
import tensorflow.keras as K
import networkAux as networkAux
import cAugmentation 

def main():

    # *****************************
    # USER SETTINGS
    # *****************************
    
    # TASK SETTINGS
    task = 'Classification'         # one of 'Classification' or 'Regression'    
    nOutNodes = 9                   # number of classes (in case of classificaion) or percentiles (in case of regression)
    fldrImgTrain = 'path/to/image/folder/training'      # folder containing all training images
    fldrImgValid = 'path/to/image/folder/validation'    # folder containing all validation images
    fldrLabel = 'path/to/reference/folder'              # folder containing txt files with reference class (single integer) in case of classification or list of percentile values in case of regression  
    
    
    # TRAINING SETTINGS
    doMultiscale = True             # if true, multi-scale encoder modules are used (cf. related publication)
    regularisation = regularisation=K.regularizers.l2(1e-5)
    learn_rate = 0.001              # learning rate
    nEpochs = 400                   # number of training epochs
    nTrainImages = 1000             # number of training images
    nValidImages = 1000             # number of validation images       
    batchsize = 8                   # batchsize
    in_size=(550.750)               # image size in order (columns, rows). The data in the input folder is required to have exactly that size
    stepsPerEpoch = nTrainImages/batchsize
    val_steps = nValidImages/batchsize    
    
    
    # *****************************
    # NETWORK AND TRAINING
    # *****************************
    
    # NETWORK
    model=networkAux.GranulometryNet(nOutNodes=nOutNodes, input_depth=3, regularisation = regularisation, doBatchnorm = False, doMultiScale = doMultiscale)    
    model.summary()
    
    # AUGMENTATION
    Augmenter = cAugmentation.cAugmentation(doRot90=False, doFlipUD=True, doFlipLR=True,
                                                brightnessShift = 0.3, contrastShift = 0.2, colorShift=20, saturationShift=20, gammaCorrection=0.2, colorSwap=0.0)
        
    if task == 'Regression':
        model.compile(loss=K.losses.KLDivergence(), optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate))
        train_generator = networkAux.generator_regression(fldrImgTrain, fldrLabel, batchsize, nOutNodes, in_size=in_size, Augmenter=Augmenter)
        valid_generator = networkAux.generator_regression(fldrImgValid, fldrLabel, batchsize, nOutNodes, in_size=in_size)
    elif task == 'Classification':
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), metrics=['acc'])
        train_generator = networkAux.generator_classification(fldrImgTrain, fldrLabel, batchsize, nOutNodes, in_size=in_size, Augmenter=Augmenter)
        valid_generator = networkAux.generator_classification(fldrImgValid, fldrLabel, batchsize, nOutNodes, in_size=in_size)
        
    
    # TRAIN
    history = model.fit_generator(train_generator, 
                                  steps_per_epoch=stepsPerEpoch,
                                  epochs = nEpochs,
                                  callbacks = [networkAux.checkpointer, networkAux.reduce_lr],
                                  validation_data=valid_generator,
                                  validation_steps=val_steps,
                                  verbose=1
                                  )

# ============================================================================================
if __name__ == "__main__":
    main()