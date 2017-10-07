import cv2
import numpy as np
from matplotlib import pyplot as plt


def convolve_1d(image,k,dft_type):
    P=0.0j

    if dft_type ==0:
        sign = -1
        scale = 1
    else:
        sign = 1
        scale = len(image)
        
    for a in range(0,len(image)):
        P = P+(image[a]*(np.exp((sign)*(1j)*2*(np.pi)*(float(k*a)/len(image)))))

    P= P/scale
    return P


def convolve_2d(image,kernel):
    #print(image)
    M=len(image)#rows
    N=len(image[0])#collumns
    a=len(kernel)
    pad_size=int(np.floor(a/2))




    padded_image=np.lib.pad(image, (pad_size,pad_size), 'constant', constant_values=(0))

    conv_img = np.zeros(shape=(M,N))
    temp_mat = np.zeros(shape=(a,a))

   
    for k in range(0,M):
        for l in range(0,N):
            temp_mat=padded_image[k:k+a,l:l+a]*kernel
            conv_img[k,l]=sum(sum(temp_mat))

    return conv_img

def getmse(orig,recon):
    mse =0
    M=len(orig)
    N=len(orig[0])
    for k in range(0,M):
        for l in range(0,N):
            mse = (np.abs((orig[k][l]-recon[k][l])))**2
    return mse

def createCrackEdges(image,T1):
    M=len(image)#rows
    N=len(image[0])#collumns
    upimage = np.zeros(shape=(M*2,N*2))

    for i in range(0,M):
        for j in range(0,N):
            upimage[i*2,j*2] = image[i,j]


        for k in range(0,M):
            for l in range(0,N):
                if k==M-1:
                    upimage[(2*k)+1,2*l]=upimage[2*k,2*l]
                else:
                    diff=abs(upimage[(2*k),2*l]-upimage[(2*k)+2,2*l])
                    if diff > T1:
                        upimage[(2*k)+1,2*l]=255
                    else:
                        upimage[(2*k)+1,2*l]=0

        for l in range(0,N):
            for k in range(0,M):
                if l==N-1:
                    upimage[2*k,(2*l)+1]=upimage[2*k,(2*l)]
                else:
                    diff=abs(upimage[2*k,(2*l)]-upimage[2*k,(2*l)+2])
                    if diff > T1:
                        upimage[2*k,(2*l)+1]=255
                    else:
                        upimage[2*k,(2*l)+1]=0
                        


    return upimage

def removeWeakEdges(image,T2,T3):
    M=int(len(image)/2)#rows
    N=int(len(image[0])/2)#collumns
    downimg = np.zeros(shape=(M,N))
    T2*=100

    change =1
    i=0
    while change == 1:
        print (i)
#        change = 0
        changeCount=0
        for k in range(0,M-1):
            for l in range(0,N-1):
                if image[(2*k),2*l]!=image[(2*k),(2*l)+2]:
                    diff1=abs(image[(2*k),2*l]-image[(2*k),(2*l)+2])
                    avg=(image[(2*k),2*l]+image[(2*k),(2*l)+2])/2         
                    if diff1 < T2 :
                        #change = 1
                        changeCount+=1
                        image[(2*k),2*l]=avg
                        image[(2*k),(2*l)+2] = avg
                        if ((image[(2*k),(2*l)+2] == 255) and (image[(2*k),2*l] == 255))==False: 
                            image[(2*k),(2*l)+1] = 0
                if image[(2*k),2*l]!=image[(2*k)+2,(2*l)]:
                    diff2=abs(image[(2*k),2*l]-image[(2*k)+2,(2*l)])
                    avg=(image[(2*k),2*l]+image[(2*k)+2,(2*l)])/2
                    if diff2 < T2:
                        #change = 1
                        changeCount+=1
                        image[(2*k),2*l]=avg
                        image[(2*k)+2,(2*l)] = avg
                        if ((image[(2*k),2*l]==255) and (image[(2*k)+2,(2*l)]==255))==False:
                            image[(2*k)+1,(2*l)] = 0

        if changeCount<T3 :
            change = 0
        else:
            change = 1

  

        i=i+1
        if (i%500==0):
            file = "seg_in_prog_iteration_%s.jpg"%(i)
            cv2.imwrite(file,image)
            downimg=image[::2,::2]
            file1 = "seg_%s.jpg"%(i)
            cv2.imwrite(file1,downimg)
            

    return image


print ("start")
img = cv2.imread('C:/Users/presh/Google Drive/MS/CVIP/HW/Hw4/MixedVegetables.jpg')
##img=cv2.equalizeHist(img)
img=cv2.medianBlur(img,5)
img=cv2.medianBlur(img,5)


##mask = np.zeros(img.shape, dtype=np.uint8)
##roi_corners = np.array([[(100,100), (100,200),(250,250), (300,150), (100,300)]], dtype=np.int32)
##channel_count = img.shape[2]
##ignore_mask_color = (255,)*channel_count
##cv2.fillPoly(mask, roi_corners, ignore_mask_color)
##seg = cv2.bitwise_and(255, mask)
##
##
##plt.subplot(122),plt.imshow(seg, cmap = 'gray')
##plt.show()
##cv2.imwrite('blur2.jpg',img)

gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##cv2.imwrite('gray.jpg',gimg)
gimg=cv2.equalizeHist(gimg)
gimg=cv2.medianBlur(gimg,5)
gimg=cv2.medianBlur(gimg,5)
M=len(gimg)#rows
N=len(gimg[0])#collumns
T1=15
T2=0.1
T3=max(1,int((4*M*N)/200))

#gimg=cv2.blur(gimg,(5,5))
#gimg=cv2.medianBlur(gimg,5)

crakedImg=createCrackEdges(gimg,T1)
cv2.imwrite('crackedImg15h.jpg',crakedImg)

finImg=removeWeakEdges(crakedImg,T2,T3)
cv2.imwrite('finImg.jpg',finImg)



    

print("done")             
