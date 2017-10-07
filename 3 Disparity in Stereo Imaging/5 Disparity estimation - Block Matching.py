import numpy as np
import cv2
import scipy as sp
from matplotlib import pyplot as plt
from scipy import ndimage


WindowSize = 9


def SSD(y,x,z,paddedImage3,paddedImage4,border):
    error = 0
    for m in np.arange(y-border,y+border+1):
        n=x-border
        p=z-border
        while(n<=x+border):
            error=error+abs(int(paddedImage3[m,n])-int(paddedImage4[m,p]))**2
            n=n+1
            p=p+1
    return error
            
def getmse(orig,recon):
    mse =0.0
    M=len(orig)
    N=len(orig[0])
    for k in range(0,M):
        for l in range(0,N):
            mse = mse+(np.abs((orig[k][l]-recon[k][l])))**2
    return mse/(M*N)

        

def Disparity1left(y,x,paddedImage3,paddedImage4,output1,border):
    error=SSD(y,x,x,paddedImage3,paddedImage4,border)
    output1[y-border,x-border]=0
    for m in np.arange(x,x+62):
        if(m>=border+imgWidth):
            break
        temperror=SSD(y,x,m,paddedImage3,paddedImage4,border)
        if(temperror<error):
            output1[y-border,x-border]=m-x
            error=temperror


def Disparity1right(y,x,paddedImage3,paddedImage4,output2,border):
    error=SSD(y,x,x,paddedImage3,paddedImage4,border)
    output2[y-border,x-border]=0
    for m in np.arange(x,x-62,-1):
        if(m<0):
            break
        temperror=SSD(y,x,m,paddedImage3,paddedImage4,border)
        if(temperror<error):
            output2[y-border,x-border]=x-m
            error=temperror

#img 1 is assigned view3 and img2 is assigned view1
img1 = sp.ndimage.imread('/Users/Rakshit/Google Drive/1st sem MS/CVIP/CVIPFinalProject/Data/view3.png',flatten=True)
img2 = sp.ndimage.imread('/Users/Rakshit/Google Drive/1st sem MS/CVIP/CVIPFinalProject/Data/view1.png',flatten=True)

#Get the height and width of image
(imgHeight, imgWidth) = img1.shape[:2]



#Create a padded images for SSD matching
border = int((WindowSize - 1) / 2)
paddedImage1 = cv2.copyMakeBorder(img1,border,border,border,border,cv2.BORDER_REPLICATE)
paddedImage2 = cv2.copyMakeBorder(img2,border,border,border,border,cv2.BORDER_REPLICATE)


#Allocate output disparity images		    	    
output1 = np.zeros((imgHeight, imgWidth), dtype="int")
output2 = np.zeros((imgHeight, imgWidth), dtype="int")
#Calculate disparity image
(padimgHeight, padimgWidth) = paddedImage1.shape[:2]
for y in np.arange(border, border+imgHeight):
    print (y)
    for x in np.arange(border, border+imgWidth):
            Disparity1left(y,x,paddedImage1,paddedImage2,output1,border)
            Disparity1right(y,x,paddedImage2,paddedImage1,output2,border)

        

plt.subplot(122),plt.imshow(output1, cmap = 'gray')
plt.show()
sp.misc.imsave('/Users/Rakshit/Google Drive/1st sem MS/CVIP/CVIPFinalProject/Data/SSD_window_size_9_right.png',output1)
sp.misc.imsave('/Users/Rakshit/Google Drive/1st sem MS/CVIP/CVIPFinalProject/Data/SSD_window_size_9_left.png',output2)


disp1=sp.ndimage.imread('/Users/Rakshit/Google Drive/1st sem MS/CVIP/CVIPFinalProject/Data/disp1.png',flatten=True)
mse=getmse(disp1,output1)
print("MSE with DP_right:",mse)
mse=getmse(disp1,output2)
print("MSE with DP_left:",mse)
    
    
    
