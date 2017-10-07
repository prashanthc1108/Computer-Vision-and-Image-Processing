import numpy as np
from matplotlib import pyplot as plt
import scipy as sp


def getmse(orig,recon):
    mse =0.0
    M=len(orig)
    N=len(orig[0])
    for k in range(0,M):
        for l in range(0,N):
            mse = mse+ (np.abs((orig[k][l]-recon[k][l])))**2
    return mse/(M*N)


img1 = sp.ndimage.imread('C:/Users/presh/Google Drive/MS/CVIP/Proj/view1.png',flatten=True)
img2 = sp.ndimage.imread('C:/Users/presh/Google Drive/MS/CVIP/Proj/view3.png',flatten=True)



#Get the height and width of image
imgHeight = len(img1)
imgWidth = len(img1[0])

C = np.zeros((imgWidth,imgWidth))
M = np.zeros((imgWidth,imgWidth))
DispL = np.zeros((imgHeight,imgWidth))
DispR = np.zeros((imgHeight,imgWidth))
occ = 20

for row in range(0,imgHeight):
    print(row),
    for i in range(1,imgWidth):
        C[i,0] = i*occ
    for j in range(1,imgHeight):
        C[0,j] = j*occ

    for i in range(1,imgWidth):
        for j in range(1,imgWidth):
            temp = int(abs(img1[row,i] - img2[row,j]))
            min1 = C[i-1,j-1]+temp
            min2 = C[i-1,j]+occ
            min3 = C[i,j-1]+occ
            cmin = min(min1,min2,min3)
            C[i,j] = cmin
            if (cmin == min1):
                M[i,j]=1
            elif (cmin == min2):
                M[i,j]=2
            elif (cmin == min3):
                M[i,j]=3
    i = imgWidth-1
    j = imgWidth-1

    while(i>0 and j>0):
        
        if (M[i,j]==1):
            DispR[row,j] = abs(j-i)
            DispL[row,i] = abs(i-j)
            i = i-1
            j = j-1
        elif (M[i,j]==2):
            DispL[row,i] = 0
            i = i-1
        elif (M[i,j]==3):
            DispR[row,j] = 0
            j=j-1

    
    C = np.zeros((imgWidth,imgWidth))
    M = np.zeros((imgWidth,imgWidth))   

plt.subplot(122),plt.imshow(DispR, cmap = 'gray')
plt.show()

sp.misc.imsave('C:/Users/presh/Google Drive/MS/CVIP/Proj/Submission/Result/DP_R.png',DispR)
sp.misc.imsave('C:/Users/presh/Google Drive/MS/CVIP/Proj/Submission/Result/DP_L.png',DispL)

disp1 = sp.ndimage.imread('C:/Users/presh/Google Drive/MS/CVIP/Proj/disp1.png',flatten=True)
mse=getmse(DispR,disp1)
print("MSE with DP_R:",mse)

mse=getmse(DispL,disp1)
print("MSE with DP_L:",mse)


