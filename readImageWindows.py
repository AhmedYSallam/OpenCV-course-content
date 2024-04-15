
# import required modules 
import cv2 
import matplotlib.pyplot as plt 

# read the image 
image = cv2.imread('C:\\Users\\Ahmed\Desktop\\Documents\\PROGRAMMING\\PythonDev\\src\\Content\\Images\\car.jpg') 
  
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
print(image.shape)
plt.imshow(image, cmap= "gray") 
cv2.imshow("frame", image)
  
# display that image 
plt.show() 