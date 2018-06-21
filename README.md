# 2D convolution using CUDA

implementation of 2D convolution 

All project include a Gaussian filter kernal(standard deviation can be adjusted it also support large standard deviation) and 5X5 vertical Sobel filter kernels. Other filters can be added. 

The program requires image with PPM format. Please use the provided python file convert_from_jpg_to_ppm.py to convert your image for processing and use
convert_from_ppm_to_jpg.py to convert the result back to JPEG format.

Requirment:

CUDA 9.1, visual Studio 2015, Cmake 3.10.3 to compile

Importent:
If your results are black please change the maskRows and maskColms to 9 instead of 11. this is because of some GPU limitation.
