// Sources: 
//  https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
//  https://dl.acm.org/doi/10.1145/2461912.2461984#sec-supp
// Code samples referenced:
//  5_Domain_Specific\fluidsGL
//  5_Domain_Specific\smokeParticles

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cudart_platform.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <iostream>
#include <math.h>

using std::cout;
using std::endl;
using std::swap;
using std::min;
using std::max;

// Structure for custom vector operators for 2D vectors of floats
struct Vec2f {
	float x = 0.0f, y = 0.0f;

	__device__ Vec2f operator+(Vec2f add) {
		return {
			add.x = this->x + add.x,
			add.y = this->y + add.y
		};
	}

	__device__ Vec2f operator-(Vec2f subtract) {
		return {
			subtract.x = this->x - subtract.x,
			subtract.y = this->y - subtract.y
		};
	}

	__device__ Vec2f operator*(float multiply) {
		Vec2f result;
		result.x = this->x * multiply;
		result.y = this->y * multiply;
		return result;
	}

	__device__ Vec2f operator/(float divide) {
		Vec2f result;
		result.x = this->x / divide;
		result.y = this->y / divide;
		return result;
	}
};

// Structure for custom color operators for 3D vectors of floats
struct Color3f {
	float R = 0.0f, G = 0.0f, B = 0.0f;

	__host__ __device__ Color3f operator+(Color3f add) {
		//Color3f result;
		return {
			add.R = this->R + add.R,
			add.G = this->G + add.G,
			add.B = this->B + add.B
		};
	}

	__host__ __device__ Color3f operator-(Color3f subtract) {
		return {
			subtract.R = this->R - subtract.R,
			subtract.G = this->G - subtract.G,
			subtract.B = this->B - subtract.B
		};
	}

	__host__ __device__ Color3f operator*(float multiply) {
		Color3f result;
		result.R = this->R * multiply;
		result.G = this->G * multiply;
		result.B = this->B * multiply;
		return result;
	}

	__host__ __device__ Color3f operator/(float divide) {
		Color3f result;
		result.R = this->R / divide;
		result.G = this->G / divide;
		result.B = this->B / divide;
		return result;
	}
};

struct Particle {
	Vec2f vel;
	Color3f color;
};

static struct Data {
	int radius = 250; // Radius of the emitter
	float vorticity = 50.0f; // Controls rotation (Higher is more chaotic)
	float velocityDiffusion = 0.75f; // Number of trails/spikes (Higher adds more)
	float colorDiffusion = 0.5f; // Color resolution (Higher adds less resolution)
	float densityDiffusion = 0.025f; // Duration (Higher dissipates slower)
	float forceScale = 2500.0f; // Emission force
	float pressure = 1.25f;
} data;

static struct GPUData {
	int xThreads = 512;
	int yThreads = 2;
	int velocitySize = 25;
	int pressureSize = 25;
} gpuData;

// Globals
static const int colorSize = 7;
Color3f colorArray[colorSize];

static Particle *curField;
static Particle *newField;

static unsigned char* colorField;

static float* curPressure;
static float* newPressure;

static float* vorticityField;

static size_t xSize, ySize;

static Color3f curColor;

static float elapsedTime = 0.0f;