
import cv2
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
img = cv2.imread("test.jpg")
def rotate_image(image):
  image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
  #image = cv2.rotate(src, cv2.ROTATE_180)
  #image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
  return image
def crop_image(image):
  y = 100
  x = 100
  h = 30
  w = 10000
  crop = image[y:y + h, x:x + w]
  return crop
def mirror_image(image):
  # cv2 format to PIL format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  im_mirror = ImageOps.mirror(image)
  return im_mirror
def flip_image(image):
  # cv2 format to PIL format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  im_flip = ImageOps.flip(image)
  return im_flip
def change_brightness_of_image(image):
  # cv2 format to PIL format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  enhancer = ImageEnhance.Brightness(image)

  factor = 0.6  # brightens the image
  im_output = enhancer.enhance(factor)
  return  im_output
def invert_image(image):
  imagem = cv2.bitwise_not(image)
  return imagem
def histogram_normalization(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  hist, bins = np.histogram(image.flatten(), 256, [0, 256])
  cdf = hist.cumsum()
  cdf_normalized = cdf * float(hist.max()) / cdf.max()
  plt.plot(cdf_normalized, color='b')
  plt.hist(image.flatten(), 256, [0, 256], color='r')
  plt.xlim([0, 256])
  plt.legend(('cdf', 'histogram'), loc='upper left')
  plt.show()
  equ = cv2.equalizeHist(image)
  #if we dont want to see gray level image, we can do splitting rgb operation
  #and apply histogram equalization
  """R, G, B = cv2.split(img)

  output1_R = cv2.equalizeHist(R)
  output1_G = cv2.equalizeHist(G)
  output1_B = cv2.equalizeHist(B)

  equ = cv2.merge((output1_R, output1_G, output1_B))"""
  cv2.imshow('equ.png', equ)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  hist, bins = np.histogram(equ.flatten(), 256, [0, 256])
  cdf = hist.cumsum()
  cdf_normalized = cdf * float(hist.max()) / cdf.max()
  plt.plot(cdf_normalized, color='b')
  plt.hist(equ.flatten(), 256, [0, 256], color='r')
  plt.xlim([0, 256])
  plt.legend(('cdf', 'histogram'), loc='upper left')
  plt.show()
  return equ
def clahe(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=40)
  gray_img_clahe = clahe.apply(image)
  cv2.imshow("Images", gray_img_clahe)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return gray_img_clahe



img = clahe(img)
#PIL format to cv2 format
#img = np.array(img)
#img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imwrite("imgtest.jpg",img)