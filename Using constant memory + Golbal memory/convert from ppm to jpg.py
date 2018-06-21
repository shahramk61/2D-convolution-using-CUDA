from PIL import Image


im = Image.open("output_gaussian.ppm")
im.save("output_gaussian.jpg")

im = Image.open("output_Sobel.ppm")
im.save("output_Sobel.ppm.jpg")


