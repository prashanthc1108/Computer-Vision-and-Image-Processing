import cv2
import numpy as np
from matplotlib import pyplot as plt


def dft_1d(image,k,dft_type):
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


def dft_2d(image,dft_type):
    #print(image)
    M=len(image)#rows
    N=len(image[0])#collumns
    G=[]
    P=[]

    P=np.zeros(shape=(M,N))*1j
    G=np.zeros(shape=(M,N))*1j

   
    for k in range(0,M):
        for l in range(0,N):
            P[k][l]=dft_1d(np.transpose(image)[l],k,dft_type)
    print("m,n",len(P),len(P[0]))
    Q=np.transpose(P)
    
    for k in range(0,M):
        for l in range(0,N):
            G[k][l]=dft_1d(np.transpose(Q)[l],k,dft_type)
    print("m,n",len(G),len(G[0]))
    F=np.transpose(G)

    return F

def getmse(orig,recon):
    mse =0
    M=len(orig)
    N=len(orig[0])
    for k in range(0,M):
        for l in range(0,N):
            mse = (np.abs((orig[k][l]-recon[k][l])))**2
    return mse


print ("start")
img = cv2.imread('C:/Users/presh/Google Drive/MS/CVIP/HW/Hw2/lena.jpg')
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
M=len(gimg)#rows
N=len(gimg[0])#collumns

dft_img = dft_2d(gimg,0)
idft_img = dft_2d(dft_img,1)
mse=getmse(idft_img,gimg)
print("MSE:",mse)

L=20*np.log(np.absolute(dft_img))

T=np.zeros(shape=(M,N))

T[0:np.floor(M/2),0:np.floor(M/2)]=L[np.floor(M/2):M,np.floor(M/2):M]
T[0:np.floor(M/2),np.floor(M/2):M]=L[np.floor(M/2):M,0:np.floor(M/2)]
T[np.floor(M/2):M,0:np.floor(M/2)]=L[0:np.floor(M/2),np.floor(M/2):M]
T[np.floor(M/2):M,np.floor(M/2):M]=L[0:np.floor(M/2),0:np.floor(M/2)]


cv2.imwrite('dft_out.jpg',np.absolute(dft_img))
cv2.imwrite('scaled.jpg',L)
cv2.imwrite('shifted.jpg',T)
cv2.imwrite('inv.jpg',np.absolute(idft_img))


plt.subplot(122),plt.imshow(np.absolute(idft_img.astype(np.uint8)), cmap = 'gray')
plt.show()

print("done")             
