
import cv2
import numpy as np
import scipy.interpolate as sc
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
import math
from random import randint,uniform
img = cv2.imread("test.jpg")
def rotate_right(image):
  image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  #image = cv2.rotate(src, cv2.ROTATE_180)
  #image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
  return image
def rotate_left(image):
  #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
  #image = cv2.rotate(src, cv2.ROTATE_180)
  image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
  return image
def crop_image(image,w,h,x,y):
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
def brighntess_change(image,value):
    alpha = 1.3
    if(value<0):
        beta = 20* value
    else:
        beta = 10 * value


    image = np.clip(alpha * image + beta, 0, 255)

    return image
def contrast_change(image,value):
  F = (259(int(value*10) + 255)) / (255(259 - int(value*10)))
  image = int(int(F) * (image - 128) + 128)
  return image
def change_brightness_of_image(image,factor):
  # cv2 format to PIL format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  enhancer = ImageEnhance.Brightness(image)
  factor=factor*0.1



  #factor = 3 # brightens the image
  im_output = enhancer.enhance(factor)
  return  im_output
def change_contrast_of_image(image,factor):
  # cv2 format to PIL format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  enhancer = ImageEnhance.Contrast(image)
  factor=factor*0.1



  #factor = 3 # brightens the image
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
  #cv2.imshow("Images", gray_img_clahe)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  gray_img_clahe = cv2.cvtColor(gray_img_clahe,cv2.COLOR_RGB2BGR)
  return gray_img_clahe
def otsu_binarization(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ret, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return thresh1
def oil_painting(image):
  res = cv2.xphoto.oilPainting(image, 7, 1)
  return res
def watercolor(image):
  res = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
  return res
def bw_pencil(image):
  dst_gray, dst_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
  return dst_gray
def colored_pencil(image):
  dst_gray, dst_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
  return dst_color
def blue_shift_filter(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
def random_filter(image):
  image = cv2.absdiff(image,100)
  return image

def fisheyelast(img):
    h, w = img.shape
    midx = w // 2
    midy = h // 2
    New = np.copy(img)
    New = New.flatten()
    img = img.flatten()
    for y in range(h):
      for x in range(w):
        dx = x - midx
        dy = y - midy
        theta = math.atan2(dy, dx)
        radius = math.sqrt(dx*dx + dy*dy)
        Nradius = radius*radius / (max(midx, midy))
        newX = midx + (Nradius*math.cos(theta))
        newY = midy + (Nradius*math.sin(theta))
        if (newX > 0 and newX < w and newY > 0 and newY < h):
          New[y*w + x] = img[int(newY) * w + int(newX)]
    return np.array(New).reshape(h, w)


def twist(img, factor=15):
  h, w = img.shape
  midx = w // 2
  midy = h // 2
  New = np.copy(img)
  New = New.flatten()
  img = img.flatten()
  for y in range(h):
    for x in range(w):

      dx = (factor*math.sin(math.pi*y / 64.0))
      dy = (factor*math.cos(math.pi*x / 64.0))
      theta = math.atan2(dy, dx)
      radius = math.sqrt(dx*dx + dy*dy)
      Nradius = radius*radius / (max(midx, midy))
      newX = (x + dx)
      newY = (y + dy)
      if (newX > 0 and newX < w and newY > 0 and newY < h):
        New[y*w + x] = img[int(newY) * w + int(newX)]
  return np.array(New).reshape(h, w)

def swirl(img, koef=0):
    h, w = img.shape
    New = np.copy(img)
    midx = w // 2
    midy = h // 2
    New = New.flatten()
    for y in range(h):
      for x in range(w):
        dx = x - midx
        dy = y - midy
        a = math.atan2(dy, dx)
        r = math.sqrt(dx**2 + dy**2)
        newX = int(midx + (r*math.cos(a + math.radians(koef*r))))
        newY = int(midy + (r*math.sin(a + math.radians(r*koef))))
        if (newX >= 0 and newX < w and newY >= 0 and newY < h):
          New[y*w + x] = New[newY*w + newX]
    return np.array(New).reshape(h, w)

def frostedgalss(img):
  h, w, d = img.shape
  overlay = np.full(img.shape, 65, dtype=np.uint8)
  overlay[:, :, 2] = 168
  overlay[:, :, 1] = 204
  overlay[:, :, 0] = 215
  output = np.copy(img)
  for i in range(h*w // 32):
    x = int(randint(0, w - 1))
    y = int(randint(0, h - 1))
    color1 = (250, 206, 135)
    color = (245, 120, 110)
    endy = np.clip(y + int(uniform(h* - 0.03, h*0.03)), 0, h)
    endx = np.clip(x + int(uniform(h* - 0.03, h * 0.03)), 0, w)

    overlay = cv2.line(overlay, (y, x), (endy, endx), color, 1)

  cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
  return output

def bathroom_effect(img):

    h, w = img.shape
    zero_img = img
    print(h, w)
    # print(new_img,img[200,200])
    # print((new_img.ndim))
    # print(img.shape)
    # print(new_img.shape)
    # print(img[240,370])
    # sys.exit()
    for y in range(h - 6):
      for x in range(w - 6):
        # new_img[y,x]=
        a = ((x + (x % 32) - 16))
        b = ((y + (y % 32) - 4))
        # print(-1*(x+(x%32)-16))
        zero_img[y, x] = img[y, np.clip(a, 0, w - 1)]
        zero_img[y, x] = img[np.clip(b, 0, h - 1), x]
        #zero_img[y,x]+=img[y,x]-(max(img[y,x+2],img[y+2,x])/2)
    return zero_img


def bath2_effect(img):
  h, w = img.shape
  New = img
  midx = w // 2
  midy = h // 2
  for y in range(h):
    for x in range(w):
      dx2 = (x - midx)**2
      dy2 = (y - midy)**2
      r = math.sqrt(dx2 + dy2)
      a = math.degrees(math.atan2(dy2, dx2))

      New[y, x] = img[y, x + (int((a + r / (8)) % 64)) - 192]
  return New

def solarization(img):
    h, w = img.shape
    New = img
    for y in range(h):
      for x in range(w):
        New[y,x] = img[y,x] if (img[y,x] > (255*x)/(2*w))  else 255-img[y,x]
        #New[y, x] = 255 - img[y, x]
    print(New, np.min(New), np.max(New))

    return New


def gamma_correction(image):
  #for gama in np.arange(0.5, 4, 0.5): ## kullanıcıdan alınabilir
    gama = 2.5
    print(gama)
    ingama = 1.0 / gama
    table = np.array([((i / 255.0)**gama) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_image = np.zeros(image.shape, image.dtype)
    new_image = np.clip(((image/ 255)**(1 / gama)) * 255, 0, 256)
    # cv2.imshow('firstgama',new_image)
    # cv2.waitKey(5000)

    image = cv2.LUT(image, table)
    #cv2.imshow(str(gama), image) ##gama için kullanılcak
    #cv2.imshow('tstgama', new_image)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    return image

def logtransform(img):
    c = 255 / math.log(256, 10)
    c = 1
    h, w, d = img.shape
    r, g, b = cv2.split(img)
    for ch in (r, g, b):
      for y in range(h):
        for x in range(w):
          f = ch[y, x]
          f = round(c * math.log(float(1 + f), 10))
          ch[y, x] = f
    cv2.imshow('log', np.dstack((r, g, b)))
    cv2.waitKey(0)
def pencil_sketch(img):
  # Convert the image into grayscale image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur the image using Gaussian Blur
  gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)

  # Convert the image into pencil sketch
  cartoon = cv2.divide(gray, gray_blur, scale=250.0)
  return cartoon

def histogram(img):
      h, w = img.shape
      his = np.arange(0, 256)*0

      for j in range(h):
        for i in range(w):
          his[img[j, i]] += 1
      cdf = np.arange(0, 256)*0
      px = np.arange(0, 256)*0
      for i in range(256):
        px[i] = his[i] / (h*w)
      for i in range(1, 256):
        his[i] = his[i - 1] + his[i]
      for j in range(h):
        for i in range(w):
          img[j, i] = round((his[img[j, i]] / (h*w))*255 )

      return img
def detail_enhancement(img):
  # convert the image into grayscale image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur the grayscale image with median blur
  gray_blur = cv2.medianBlur(gray, 3)

  # Apply adaptive thresholding to detect edges
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

  # Sharpen the image
  color = cv2.detailEnhance(img, sigma_s=5, sigma_r=0.5)

  # Merge the colors of same images using "edges" as a mask
  cartoon = cv2.bitwise_and(color, color, mask=edges)
  return cartoon
def bilateral(img):
  # Convert image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply median blur
  gray = cv2.medianBlur(gray, 3)

  # Detect edges with adaptive threshold
  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

  # Apply bilateral filter
  color = cv2.bilateralFilter(img, 5, 50, 5)

  # Merge the colors of same image using "edges" as a mask
  cartoon = cv2.bitwise_and(color, color, mask=edges)
  return cartoon
def negative_filter(img):
  imagem = cv2.bitwise_not(img)
  return imagem

def sharpen(image):
  kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  return cv2.filter2D(image, -1, kernel)


def sepia(image):
  kernel = np.array([[0.1, 0.3, 0.131],
                     [0.1, 0.3, 0.168],
                     [0.1, 0.3, 0.189]])
  return cv2.filter2D(image, -1, kernel)

def gaussianBlur(image):
  return cv2.GaussianBlur(image, (15, 15), 0)

def emboss(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    return cv2.filter2D(image, -1, kernel)
def spreadLookupTable(x, y):
  spline = sc.UnivariateSpline(x, y)
  return spline(range(256))

def coldImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

def warmImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))
def black_white(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  return blackAndWhiteImage
def gray(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return gray
def edges_filter(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  img1 = image.filter(FIND_EDGES)
  img = np.array(img1)
  return img

def random_filter_2(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 30, 90, 255])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 200])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


def random_filter_3(image):
  increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [20, 100, 100, 256])
  decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [20, 100, 100, 256])
  red_channel, green_channel, blue_channel = cv2.split(image)
  red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
  blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
  return cv2.merge((red_channel, green_channel, blue_channel))


img = change_brightness_of_image(img,0.5)
#PIL format to cv2 format
img = np.array(img)
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imwrite("imgtest.jpg",img)