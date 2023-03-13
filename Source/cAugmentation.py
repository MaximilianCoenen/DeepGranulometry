import numpy as np
import cv2 as cv
import random

class cAugmentation:
    """
    Class used for data augmentation
    """
    def __init__(self,
                 doRot90=False, doFlipUD = False, doFlipLR = False,
                 brightnessShift=0.0, contrastShift=0.0, colorShift = 0.0, saturationShift=0.0, gammaCorrection=0.0, colorSwap=0.0):
        """
        Parameters
        ----------
        doRot90 : bool
            if true: random rotations by 90 degree
        doFlipUD : bool
            if true: random horizontal flips
        doFlipLR : bool
            if true: random vertical flips
        brightnessShift: float [0,1]
             factor for degree of random brightness change: 0=no change, 1=strong changes
        contrastShift : float [0,1]
            offest by which the color channels are shifted
        colorShift : int [0,180]
            random shifts in the hue space in range of specified amount
        saturationShift : int [0,255]
            random shifts in saturation space in range of specified amount
        gammaCorrection : float [0,1]
            random gamma correction with gamma = 1 +- rand(in_gamma)
        colorSwap : float [0,1]
            probability for swaping the color channel of the input image
        """
        self.geometricAug = dict(flipH = doFlipUD, 
                                 flipV = doFlipLR,
                                 rot90 = doRot90)
        self.radiometricAug = dict(brightness=brightnessShift,
                                   contrast=contrastShift,
                                   hueShift=colorShift,
                                   satShift=saturationShift,
                                   gammaCorr=gammaCorrection,
                                   probColorSwap=colorSwap)
    
    #_________________________________________________________________________
    def augment_data(self, img, labelList=[], oriMap=np.array([])):
        """Augments the input by the specified parameters of self

        Parameters
        ----------
        img : numpy array of type float32 [0,1]
            the input image that is to be augmented        
            
        Returns
        ----------
        x : numpy array of type float32 [0,1]
            the augmented image        
        """
        if img.dtype != np.float32:
            raise ValueError("Error in augment_data: Input must be of type float32")
        if (np.amax(img) > 1.0 or np.amin(img)<0.0):    
            raise ValueError("Error in augment_data: Input must be of type float32 in range [0,1]")
            
        x=img.copy()      
        
        # ====================================================================
        # GEOMETRIC AUGMENTATIONS
        # ====================================================================
        x = self.augment_data_geometric(x)
            
        # ====================================================================
        # Radiometric AUGMENTATIONS
        # ====================================================================        
        x = self.augment_data_radiometric(x)
                
        return x
    
    
    
    #_________________________________________________________________________
    def augment_data_geometric(self, x):
        # Random flips (horizontal axis)
        if self.geometricAug.get('flipH'):
            x,boolFlipUD=self.rand_flipud(x)
            
        # Random flips (vertical axis)
        if self.geometricAug.get('flipV'):
            x,boolFlipLR=self.rand_fliplr(x)
        
        # Random rotations by factor*90[deg]
        if self.geometricAug.get('rot90'):
            x,nRot=self.rand_rotate90(x)
        
        return x
    
    #_________________________________________________________________________
    def augment_data_radiometric(self, x):
        imgdim = x.shape
        
        # brightness and contrast shift
        if (self.radiometricAug.get('brightness') > 0.0 or self.radiometricAug.get('contrast') > 0.0):
            alpha = 1.0 + random.uniform(-self.radiometricAug.get('brightness'), self.radiometricAug.get('brightness'))
            beta = random.uniform(-self.radiometricAug.get('contrast'), self.radiometricAug.get('contrast'))
            x = x * alpha + beta       
            x[x>1.0] = 1.0
            x[x<0.0] = 0.0       
            
        # gamma correction
        if self.radiometricAug.get('gammaCorr') > 0.0:
            gamma = 1.0 + random.uniform(-self.radiometricAug.get('gammaCorr'), self.radiometricAug.get('gammaCorr'))
            x=np.power(x,gamma)
        
        # hsv operations
        if (self.radiometricAug.get('hueShift')> 0 or self.radiometricAug.get('satShift') > 0):
            x=self.rand_hsv_operations(x)
            
        # color channel swap    
        if (self.radiometricAug.get('probColorSwap') > 0.0 and len(imgdim)==3):
            if (imgdim[2]==3):
                x = self.rand_swapChannels(x)
        
        return x
       
            
    #_________________________________________________________________________
    def rand_rotate90(self, img):
        nRot = np.random.randint(0,3)
        rotations = ['NoROTATION', cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_CLOCKWISE]
        x=img
        
        if nRot >0:
            #x = np.rot90(img, k=nRot, axes=(0,1))
            x = cv.rotate(x, rotations[nRot])           
            
        return x, nRot
        
    #_________________________________________________________________________
    def rand_flipud(self, img):
        x=img        
        
        boolFlip = True if np.random.random_sample() > 0.5 else False
        if boolFlip:
            x = np.flipud(img)
            
        return x, boolFlip
    
    #_________________________________________________________________________
    def rand_fliplr(self, img):
        x=img        
        
        boolFlip = True if np.random.random_sample() > 0.5 else False
        if boolFlip:
            x = np.fliplr(img)
            
        return x, boolFlip
    
    #_________________________________________________________________________
    def rand_swapChannels(self, img):
        x=img
        if np.random.random_sample() > self.radiometricAug.get('probColorSwap'):
            return x
        nSwapCase = np.random.randint(0,3)
        if nSwapCase ==0:
            x[:,:,[0,1]] = x[:,:,[1,0]]
        elif nSwapCase == 1:
            x[:,:,[0,2]] = x[:,:,[2,0]]
        elif nSwapCase == 2:
            x[:,:,[1,2]] = x[:,:,[2,1]]
        return x
    
    #_________________________________________________________________________
    def rand_hsv_operations(self, img):
        x=img
        x=(x*255.).astype(np.uint8)
        x=cv.cvtColor(x, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(x)
        
        # hue
        if self.radiometricAug.get('hueShift') > 0:
            shift_h = np.random.randint(-self.radiometricAug.get('hueShift'), self.radiometricAug.get('hueShift'))
            h = h.astype('int16') + shift_h
            h[h>180]-=180
            h[h<0]+=180
            h = h.astype('uint8')
        
        # sat
        if self.radiometricAug.get('satShift') > 0:
            shift_s = np.random.randint(-self.radiometricAug.get('satShift'), self.radiometricAug.get('satShift'))
            s = s.astype('int16') + shift_s
            s[s>255] = 255
            s[s<0] = 0
            s=s.astype('uint8')
        
        shift_hsv = cv.merge([h, s, v])
        x=cv.cvtColor(shift_hsv, cv.COLOR_HSV2BGR)
        x=x.astype(np.float32)/255.
        return x
    