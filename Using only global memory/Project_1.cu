#include <iostream>
#include <cstdlib>
#include <time.h>
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>
#include "Utils.h"
using namespace std;
using namespace std::chrono;


#define maskCols 11//size of the gaussian filter column
#define maskRows 11//size of the gaussian filter row
#define SobelCols 5//size of the Sobel filter column
#define SobelRows 5//size of the Sobel filter row




//Gaussian filter value generator
void FilterCreation(float GKernel[][maskCols],float sigma)
{
	
	float r, s = 2.0 * sigma * sigma;
	float TEMP = 0;

	// sum is for normalization
	float sum = 0.0;
	int size = maskCols / 2;

	TEMP = (exp(-(1) / s)) / (3.14159265358979 * s);

	for (int x = -1 * size; x <= size; x++)
	{
		for (int y = -1 * size; y <= size; y++)
		{
			r = sqrt(x*x + y*y);
			GKernel[x + size][y + size] =
				(exp(-(r*r) / s)) / (3.14159265358979 * s);
			GKernel[x + size][y + size] = GKernel[x + size][y + size] / TEMP;
			sum += GKernel[x + size][y + size];
		}
	}

	// normalising the Kernel
	for (int i = 0; i < maskCols; ++i)
		for (int j = 0; j < maskCols; ++j)
			GKernel[i][j] /= sum;
}


//swap pointers
void swap(float* &a, float* &b) {

	float *temp = a;
	a = b;
	b = temp;

}


// Gaussian GPU kernel
__global__ void GaussianKernel(float * InputImageData, float *kernel,
	float* outputImageData, int channels, int width, int height) {


	
	int col = threadIdx.x + blockIdx.x * blockDim.x;   
	int row = threadIdx.y + blockIdx.y * blockDim.y;   
	int maskRowsRadius = maskRows / 2;
	int maskColsRadius = maskCols / 2;
	float accum;



	for (int k = 0; k < channels; k++) {      
		if (row < height && col < width) {
			accum = 0;
			int startRow = row - maskRowsRadius; 
			int startCol = col - maskColsRadius;  

			for (int i = 0; i < maskRows; i++) { 

				for (int j = 0; j < maskCols; j++) { 

					int currentRow = startRow + i;
					int currentCol = startCol + j; 

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {

						accum += InputImageData[(currentRow * width + currentCol)*channels + k] *
							kernel[i * maskRows + j];
					}
					else accum = 0;
				}

			}
			outputImageData[(row* width + col) * channels + k] = accum;
		}

	}

}


//sobel GPU kernel
__global__ void SobelKernel(float * InputImageData, float *kernel,
	float* outputImageData, int channels, int width, int height) {

	float accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int maskRowsRadius = SobelRows / 2;
	int maskColsRadius = SobelCols / 2;


	for (int k = 0; k < channels; k++) {
		if (row < height && col < width) {
			accum = 0;
			int startRow = row - maskRowsRadius;
			int startCol = col - maskColsRadius;

			for (int i = 0; i < SobelRows; i++) {
				for (int j = 0; j < SobelCols; j++) {

					int currentRow = startRow + i;
					int currentCol = startCol + j;

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {

						accum += InputImageData[(currentRow * width + currentCol)*channels + k] *
							kernel[i * SobelRows + j];
					}
					else
						accum = 0;
				}

			}
			outputImageData[(row* width + col) * channels + k] = accum;
		}

	}

}


int main() {

	float* deviceInputImageData;
	float* deviceGaussianOutput;
	float* deviceSobelOutput;
	float* deviceGaussian;
	float* deviceSobel;
	FILE* file;
	char *header;
	char *line;
	int ii, jj, kk, channels;
	int width, height, depth;
	unsigned char *charData, *charIter;
	float *floatIter;
	float scale;
	const char *line_temp;
	int iter = 0;
	int multiplier = 0;
	float hostGaussian_temp[maskRows][maskCols];
	float hostGaussian[maskRows*maskCols];

	
	//Sobel kernel
	float hostSobel[SobelRows * SobelCols] = {
		1, 2, 0, -2, -1,
		4, 8, 0, -8, -4,
		6, 12, 0, -12, -6,
		4, 8, 0, -8, -4,
		1, 2, 0, -2, -1
	};




	//****************standard deviation*********************
	//*******************************************************
	float sigma = 40;
	//*******************************************************
	//*******************************************************




	//calculate how many time to convolve(this is estimation to be close to actual STD)
	iter = (((int)round(sigma))/4)+1;
	multiplier = (((int)round(sigma)) / 5) + 1;



	//create the Gaussian filter
	FilterCreation(hostGaussian_temp,sigma/iter);


	//print the Gaussian kernel
	printf("Gaussian Kernel:\n");
	for (int i = 0; i < maskCols; ++i)
	{
		for (int j = 0; j < maskCols; ++j) {
			hostGaussian[i * maskCols + j] = hostGaussian_temp[i][j];
			printf("%.3f\t", hostGaussian_temp[i][j]);

		}
		cout << endl;
	}


	printf("\n\n\n");


	//print the Sobel kernel
	printf("Sobel Kernel:\n");
	for (int i = 0; i < SobelRows; ++i)
	{
		for (int j = 0; j < SobelCols; ++j) {
			cout << hostSobel[i * 5 + j] << "\t";
		}
		cout << endl;
	}

	printf("\n\n\n");



	//************** Input Image name****************
	//***********************************************
	file = fopen("1.ppm", "rb"); 
	//***********************************************
	//***********************************************



	if (file == NULL) {
		printf("Could not open\n");
		exit(0);
	}

	header = File_readLine(file);
	if (header == NULL) {
		printf("Could not read from\n");
		exit(0);
	}

	//read image width and height
	if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
		channels = 3;
		line = nextLine(file);
		line_temp = skipSpaces(line);
		sscanf(line_temp, "%d %d", &width, &height);
	}
	else {
		printf("Wrong file formate!");
	}

	//read image depth
	line = nextLine(file);
	line_temp = skipSpaces(line);
	sscanf(line, "%d", &depth);



	//read the image data
	charData = (unsigned char *)File_read(file, width * channels * sizeof(unsigned char), height);

	//create the input data structure
	float *data = (float*)malloc(sizeof(float) * width * height * channels);




	charIter = charData;
	floatIter = data;

	scale = 1.0f / ((float)depth);

	//put the input image data in the input data structure
	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (kk = 0; kk < channels; kk++) {
				*floatIter = ((float)*charIter) * scale;
				floatIter++;
				charIter++;
			}
		}
	}


	//create output data 
	float *data_2 = (float*)malloc(sizeof(float) * width * height * channels);//Gaussian result
	float *data_3 = (float*)malloc(sizeof(float) * width * height * channels);//Sobel result



	//allocate device memory
	cudaMalloc((void **)&deviceInputImageData, width * height *channels * sizeof(float));
	cudaMalloc((void **)&deviceGaussianOutput, width * height *channels * sizeof(float));
	cudaMalloc((void **)&deviceSobelOutput, width * height *channels * sizeof(float));
	cudaMalloc((void **)&deviceGaussian, maskRows * maskCols * sizeof(float));
	cudaMalloc((void **)&deviceSobel, SobelRows * SobelCols * sizeof(float));
	
	//copy input data to device
	cudaMemcpy(deviceInputImageData, data, width * height *channels * sizeof(float), cudaMemcpyHostToDevice);

	//copy filter kernels to constant memory
	cudaMemcpy(deviceGaussian, hostGaussian, maskRows * maskCols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSobel, hostSobel, SobelRows * SobelCols * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//calculating gride and block dimentions
	dim3 dimGrid(ceil((float)width / 16), ceil((float)height / 16));
	dim3 dimBlock(16, 16, 1);


	printf("Gaussian KERNEL CONVOLUTION using global Memory only :\n");
	printf("image dimention: %d x %d , number of iteration: %d with sigma: %.2f\n", width, height,iter*multiplier,sigma/iter);
	printf("elapsed in time: ");

	auto start_t = high_resolution_clock::now();
	auto start = high_resolution_clock::now();
	for (int i = 0; i < iter*multiplier; i++) {

		GaussianKernel << <dimGrid, dimBlock >> > (deviceInputImageData, deviceGaussian, deviceGaussianOutput, channels, width, height);
		cudaDeviceSynchronize();
		swap(deviceInputImageData, deviceGaussianOutput);


	}
	swap(deviceInputImageData, deviceGaussianOutput);
	auto end = high_resolution_clock::now();


	cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() <<" milliseconds --> done!\n\n" ;
	



	printf("Sobel KERNEL CONVOLUTION using global Memory only:\n");
	printf("elapsed in time: ");

	auto start_2 = high_resolution_clock::now();
	SobelKernel << <dimGrid, dimBlock >> >(deviceGaussianOutput, deviceSobel, deviceSobelOutput, channels, width, height);
	cudaDeviceSynchronize();
	auto end_2 = high_resolution_clock::now();
	auto end_t = high_resolution_clock::now();
	cout << chrono::duration_cast<chrono::milliseconds>(end_2 - start_2).count() << " milliseconds --> done!\n\n";
	
	cudaMemcpy(data_2, deviceGaussianOutput, width * height *channels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(data_3, deviceSobelOutput, width * height *channels * sizeof(float), cudaMemcpyDeviceToHost);

	cout<<"Total time ->"<< chrono::duration_cast<chrono::milliseconds>(end_t - start_t).count() <<" milliseconds\n\n";




	//************** Gaussian Output name************
	//***********************************************
	file = fopen("output_gaussian.ppm", "wb+");
	//***********************************************
	//***********************************************



	printf("Saving output_gaussian.ppm\n");
	if (file == NULL) {
		printf("Could not opens\n");
		return false;
	}
	fprintf(file, "P6\n");
	fprintf(file, "%d %d\n", width, height);
	fprintf(file, "%d\n", depth);

	charData = (unsigned char*)malloc(sizeof(unsigned char) *  width *  height *  channels);
	charIter = charData;
	floatIter = data_2;



	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (kk = 0; kk < channels; kk++) {
				*charIter = (unsigned char)ceil(
					clamp(*floatIter, 0, 1) * depth);
				floatIter++;
				charIter++;
			}
		}
	}


	bool writeResult = File_write(file, charData, width * channels * sizeof(unsigned char), height);
	printf("Saving done!\n");
	fclose(file);



	//************** Sobel Output name***************
	//***********************************************
	file = fopen("output_Sobel.ppm", "wb+");
	//***********************************************
	//***********************************************



	printf("Saving output_Sobel.ppm\n");
	if (file == NULL) {
		printf("Could not opens\n");
		return false;
	}
	fprintf(file, "P6\n");
	fprintf(file, "%d %d\n", width, height);
	fprintf(file, "%d\n", depth);

	charData = (unsigned char*)malloc(sizeof(unsigned char) *  width *  height *  channels);
	charIter = charData;
	floatIter = data_3;



	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (kk = 0; kk < channels; kk++) {
				*charIter = (unsigned char)ceil(
					clamp(*floatIter, 0, 1) * depth);
				floatIter++;
				charIter++;
			}
		}
	}

	writeResult = File_write(file, charData, width * channels * sizeof(unsigned char), height);
	printf("Saving done!\n");
	fclose(file);



	free(charData);
	fflush(file);


	return true;
}
