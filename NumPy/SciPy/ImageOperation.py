import imageio

# Read a JPEG image into a numpy array
img = imageio.v3.imread('t.jpg')
print(img.shape, img.dtype)

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]
# Resize the tinted image to be 300 by 300 pixels.

# Write the tinted image back to disk
imageio.imsave('assets/cat_tinted.jpg', img_tinted)