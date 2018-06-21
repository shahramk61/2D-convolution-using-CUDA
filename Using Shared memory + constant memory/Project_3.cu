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

#define TILE_WIDTH 32
#define maskCols 11//size of the gaussian filter column
#define maskRows 11//size of the gaussian filter row
#define SobelCols 5//size of the Sobel filter column
#define SobelRows 5//size of the Sobel filter row
#define w_gauss (TILE_WIDTH + maskCols -1)
#define w_sobel (TILE_WIDTH + SobelCols -1)


//Gaussian filter value generator
void FilterCreation(float GKernel[][maskCols],float sigma)
{

	float r, s = 2.0 * sigma * sigma;

	// sum is for normalization
	float sum = 0.0;
	int size = maskCols / 2;

	for (int x = -1 * size; x <= size; x++)
	{
		for (int y = -1 * size; y <= size; y++)
		{
			r = sqrt(x*x + y*y);
			GKernel[x + size][y + size] =
				(exp(-(r*r) / s)) / (3.14159265358979 * s);
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

//mask in constant memory
__constant__ float deviceSobel[SobelRows * SobelCols];
__constant__ float deviceGaussian[maskRows * maskCols];




//Gaussian GPU kernel
__global__ void GaussianKernel(float * InputImageData, const float *__restrict__ kernel,float* outputImageData, int channels, int width, int height) {

	__shared__ float N_ds[w_gauss][w_gauss];	


	int maskRadius = maskRows / 2;
	for (int k = 0; k <channels; k++) {
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int destY = dest / w_gauss;     
		int destX = dest%w_gauss;		
		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;  
		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;	
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = InputImageData[(srcY *width + srcX) * channels + k];
		else
			N_ds[destY][destX] = 0;


		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / w_gauss;
		destX = dest%w_gauss;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		if (destY < w_gauss) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				N_ds[destY][destX] = InputImageData[(srcY *width + srcX) * channels + k];
			else
				N_ds[destY][destX] = 0;
		}

		__syncthreads();


		float accum = 0;
		int y, x;
		for (y = 0; y < maskCols; y++)
			for (x = 0; x<maskRows; x++)
				accum += N_ds[threadIdx.y + y][threadIdx.x + x] * deviceGaussian[y * maskCols + x];

		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			outputImageData[(y * width + x) * channels + k] = accum;
		__syncthreads();


	}

}


//Sobel GPU kernel
__global__ void SobelKernel(float * InputImageData, const float *__restrict__ kernel,
	float* outputImageData, int channels, int width, int height) {

	__shared__ float N_ds[w_sobel][w_sobel];	


												
	int maskRadius = SobelRows / 2;
	for (int k = 0; k <channels; k++) {
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int destY = dest / w_sobel;    
		int destX = dest%w_sobel;		
		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;  
		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;	
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = InputImageData[(srcY *width + srcX) * channels + k];
		else
			N_ds[destY][destX] = 0;


		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / w_sobel;
		destX = dest%w_sobel;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		if (destY < w_sobel) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				N_ds[destY][destX] = InputImageData[(srcY *width + srcX) * channels + k];
			else
				N_ds[destY][destX] = 0;
		}

		__syncthreads();


		float accum = 0;
		int y, x;
		for (y = 0; y < SobelCols; y++)
			for (x = 0; x<SobelRows; x++)
				accum += N_ds[threadIdx.y + y][threadIdx.x + x] * deviceSobel[y * SobelCols + x];

		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			outputImageData[(y * width + x) * channels + k] = accum;
		__syncthreads();


	}

}


int main() {

	int imageChannels;
	int imageHeight;
	int imageWidth;
	float* hostInputImageData;
	float* hostGaussianOutput;
	float* deviceInputImageData;
	float* deviceGaussianOutput;
	float* deviceSobelOutput;
	float total_time = 0;
	float* device;
	FILE* file;
	char *header;
	char *line;
	int ii, jj, kk, channels;
	int width, height, depth;
	unsigned char *charData, *charIter;
	float *imgData, *floatIter;
	float scale;
	const char *line_temp;
	int iter = 0;
	int multiplier = 0;



	//****************standard deviation*********************
	//*******************************************************
	float sigma = 40;
	//*******************************************************
	//*******************************************************

	//calculate how many time to convolve 
	iter = (((int)round(sigma)) / 4) + 1;
	multiplier = (((int)round(sigma)) / 5) + 1;


	//Vertical Sobel kernal value
	float hostSobel[SobelRows * SobelCols] = {
		1, 2, 0, -2, -1,
		4, 8, 0, -8, -4,
		6, 12, 0, -12, -6,
		4, 8, 0, -8, -4,
		1, 2, 0, -2, -1
	};





	float hostGaussian_temp[maskRows][maskCols];
	float hostGaussian[maskRows*maskCols];

	//Gaussian kernel value generator
	FilterCreation(hostGaussian_temp,sigma/iter);

	printf("Gaussian Kernel:\n");
	for (int i = 0; i < maskCols; ++i)
	{
		for (int j = 0; j < maskCols; ++j) {
			hostGaussian[i * maskCols + j] = hostGaussian_temp[i][j];
			printf("%.2f\t", hostGaussian_temp[i][j]);

		}
		cout << endl;
	}

	printf("Sobel Kernel:\n");
	for (int i = 0; i < SobelRows; ++i)
	{
		for (int j = 0; j < SobelCols; ++j) {
			cout << hostSobel[i * 5 + j] << "\t";

		}
		cout << endl;
	}



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

	//copy input data to device
	cudaMemcpy(deviceInputImageData, data, width * height *channels * sizeof(float), cudaMemcpyHostToDevice);

	//copy filter kernels to constant memory
	cudaMemcpyToSymbol(deviceGaussian, hostGaussian, maskRows * maskCols * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(deviceSobel, hostSobel, SobelRows * SobelCols * sizeof(float), 0, cudaMemcpyHostToDevice);

	//calculating gride and block dimentions
	float numberBlockXTiling = (float)width / TILE_WIDTH;
	float numberBlockYTiling = (float)height / TILE_WIDTH;
	int numberBlockX = ceil(numberBlockXTiling);
	int numberBlockY = ceil(numberBlockYTiling);
	dim3 dimGrid(numberBlockX, numberBlockY);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


	printf("Gaussian KERNEL CONVOLUTION using shared and constant memory:\n");
	printf("image dimention: %d x %d, number of iteration: %d with STD: %.2f to produce %.2f SDT\n", width, height, iter*multiplier, sigma / iter,sigma);
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


	cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds --> done!\n\n";




	printf("Sobel KERNEL CONVOLUTION using shared and constant memory:\n");
	printf("image dimention: %d x %d\n", width, height);
	printf("elapsed in time: ");


	auto start_2 = high_resolution_clock::now();
	SobelKernel << <dimGrid, dimBlock >> >(deviceGaussianOutput, deviceSobel, deviceSobelOutput, channels, width, height);

	cudaDeviceSynchronize();
	auto end_2 = high_resolution_clock::now();
	auto end_t = high_resolution_clock::now();
	cout << chrono::duration_cast<chrono::milliseconds>(end_2 - start_2).count() << " milliseconds --> done!\n\n";
	cout << "Total time ->" << chrono::duration_cast<chrono::milliseconds>(end_t - start_t).count() << " milliseconds\n\n";

	cudaMemcpy(data_2, deviceGaussianOutput, width * height *channels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(data_3, deviceSobelOutput, width * height *channels * sizeof(float), cudaMemcpyDeviceToHost);




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
