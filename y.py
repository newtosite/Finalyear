#################GREY########################################################################
from PIL import Image
image = Image.open('ramsir_4.jpg')
greyscale_image = image.convert('L')
greyscale_image.save('grey.jpg');
############################GAMA###########################
#import scipy.misc
#from scipy import misc
#from scipy.misc.pilutil import Image

#im = Image.open('grey.jpg')
#im_array = scipy.misc.fromimage(im)
#im_inverse = 255 - im_array
#im_result = scipy.misc.toimage(im_inverse)
#misc.imsave('result.jpg',im_result)

#####################################DENOISING#################################################
import numpy as np
import cv2
from matplotlib import pyplot as plt
#
img = cv2.imread('grey.jpg')
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb
#
## Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
#
b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
cv2.imwrite('denLic.jpg',rgb_dst)
#plt.subplot(211),plt.imshow(rgb_img)
#plt.subplot(212),plt.imshow(rgb_dst)
#plt.show()
########################################EXTRACTING###########################################################
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

#image = Image.open('clear1.jpg')
#greyscale_image = image.convert('L')
#greyscale_image.save('greyscale_image.jpg')

#print(pytesseract.image_to_string(rgb_dst))
print(pytesseract.image_to_string(Image.open('denLic.jpg')))

foo = pytesseract.image_to_string(Image.open('denLic.jpg'))
print(foo.split(' : '))
x = open("rc.txt","w")
x.write("%s" % foo)
x.close()



