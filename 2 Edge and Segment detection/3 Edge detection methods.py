import cv2
import numpy as np
from matplotlib import pyplot as plt



def convolve_2d(image,kernel):
    #print(image)
    M=len(image)#rows
    N=len(image[0])#collumns
    a=len(kernel)
    pad_size=int(np.floor(a/2))

    padded_image=np.lib.pad(image, (pad_size,pad_size), 'constant', constant_values=(0))

#    plt.subplot(122),plt.imshow(padded_image, cmap = 'gray')
#    plt.show()

    conv_img = np.zeros(shape=(M,N))
    temp_mat = np.zeros(shape=(a,a))

   
    for k in range(0,M):
        for l in range(0,N):
            temp_mat=padded_image[k:k+a,l:l+a]*kernel
            conv_img[k,l]=sum(sum(temp_mat))

    return conv_img

def getZeroCross(image):
    M=len(image)#rows
    N=len(image[0])#collumns
    ZCrImg = np.zeros(shape=(M,N))
    for k in range(0,M):
        for l in range(0,N):
            if (k!=(M-1)) and (l!=(N-1)):
                if ((image[k,l]*image[k,l+1])<0) or ((image[k,l]*image[k+1,l])<0):
                    ZCrImg[k,l]=0
                else:
                    ZCrImg[k,l]=255
            else:
                ZCrImg[k,l]=0

    return ZCrImg

def getPrewittEdges(image):
    M=len(image)#rows
    N=len(image[0])#collumns
    p1 =([1,1,1],
         [0,0,0],
         [-1,-1,-1])
    p2 =([-1,-1,-1],
         [0,0,0],
         [1,1,1])
    p3 =([1,1,0],
         [1,0,-1],
         [0,-1,-1])
    p4 =([0,1,1],
         [-1,0,1],
         [-1,-1,0])
    a=len(p1)
    pad_size=int(np.floor(a/2))

    padded_image=np.lib.pad(image, (pad_size,pad_size), 'constant', constant_values=(0))

    res=np.zeros(shape=(8))
    conv_img = np.zeros(shape=(M,N))
    temp_mat = np.zeros(shape=(a,a))

   
    for k in range(0,M):
        for l in range(0,N):
            temp_mat=padded_image[k:k+a,l:l+a]*p1
            res[0]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*p2
            res[1]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*p3
            res[2]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*p4
            res[3]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*np.transpose(p1)
            res[4]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*np.transpose(p2)
            res[5]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*np.transpose(p3)
            res[6]=sum(sum(temp_mat))
            temp_mat=padded_image[k:k+a,l:l+a]*np.transpose(p4)
            res[7]=sum(sum(temp_mat))
            conv_img[k,l]=max(res)

    return conv_img


def removeWeakEdges(ZImage,fOImage,Threshold):
    M=len(ZImage)#rows
    N=len(ZImage[0])#collumns
    
    StrEdgeImg = np.zeros(shape=(M,N))
    for k in range(0,M):
        for l in range(0,N):
            if(fOImage[k,l]<Threshold):
                StrEdgeImg[k,l]=255
            else:
                StrEdgeImg[k,l]=ZImage[k,l]

    return StrEdgeImg

def invertC(ZImage):
    M=len(ZImage)#rows
    N=len(ZImage[0])#collumns
    newImage=np.zeros(shape=(M,N))
    
    for k in range(0,M):
        for l in range(0,N):
            newImage[k,l]=255-ZImage[k,l]

    return newImage

            

print ("start")
img = cv2.imread('C:/Users/presh/Google Drive/MS/CVIP/HW/Hw4/UBCampus.jpg')
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
M=len(gimg)#rows
N=len(gimg[0])#collumns



#gausian=([0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625])

DoG_mask=([0,0,-1,-1,-1,0,0],
          [0,-2,-3,-3,-3,-2,0],
          [-1,-3,5,5,5,-3,-1],
          [-1,-3,5,16,5,-3,-1],
          [-1,-3,5,5,5,-3,-1],
          [0,-2,-3,-3,-3,-2,0],
          [0,0,-1,-1,-1,0,0])

LoG_mask=([0,0,1,0,0],
          [0,1,2,1,0],
          [1,2,-16,2,1],
          [0,1,2,1,0],
          [0,0,1,0,0])
    
Threshold=100
firstOrderEdges = getPrewittEdges(gimg)
print("got firstOrder Edges")

DoGImage = convolve_2d(gimg,DoG_mask)
print("got DoG")
ZCrossImgDoG = getZeroCross(DoGImage)
print("got ZCross of DoG")
strngEdgImgDoG = removeWeakEdges(ZCrossImgDoG,firstOrderEdges,Threshold)
print("got Strong Edges of DoG")
InvstrngEdgImgDoG = invertC(strngEdgImgDoG)
print("got InvstrngEdgImg of DoG")

LoGImage = convolve_2d(gimg,LoG_mask)
print("got LoG")
ZCrossImgLoG = getZeroCross(LoGImage)
print("got ZCross of LoG")
strngEdgImgLoG = removeWeakEdges(ZCrossImgLoG,firstOrderEdges,Threshold)
print("got Strong Edges of LoG")
InvstrngEdgImgLoG = invertC(strngEdgImgLoG)
print("got InvstrngEdgImg of LoG")




##plt.subplot(122),plt.imshow(firstOrderEdges, cmap = 'gray')
##plt.show()
cv2.imwrite('firstOrderEdges.jpg',firstOrderEdges)

##plt.subplot(122),plt.imshow(DoGImage, cmap = 'gray')
##plt.show()
cv2.imwrite('DoGImage.jpg',DoGImage)    

##plt.imshow(ZCrossImgDoG)
##plt.show()
cv2.imwrite('ZCrossImgDoG.jpg',ZCrossImgDoG)

##plt.subplot(122),plt.imshow(strngEdgImgDoG, cmap = 'gray')
##plt.show()
cv2.imwrite('strngEdgImgDoG.jpg',strngEdgImgDoG)

##plt.subplot(122),plt.imshow(InvstrngEdgImgDoG, cmap = 'gray')
##plt.show()
cv2.imwrite('InvstrngEdgImgDoG.jpg',InvstrngEdgImgDoG)

##plt.subplot(122),plt.imshow(LoGImage, cmap = 'gray')
##plt.show()
cv2.imwrite('LoGImage.jpg',LoGImage)
   
##plt.imshow(ZCrossImgLoG)
##plt.show()
cv2.imwrite('ZCrossImgLoG.jpg',ZCrossImgLoG)

##plt.subplot(122),plt.imshow(strngEdgImgLoG, cmap = 'gray')
##plt.show()
cv2.imwrite('strngEdgImgLoG.jpg',strngEdgImgLoG)

##plt.subplot(122),plt.imshow(InvstrngEdgImgLoG, cmap = 'gray')
##plt.show()
cv2.imwrite('InvstrngEdgImgLoG.jpg',InvstrngEdgImgLoG)


print("done")             
