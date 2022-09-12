#include "Header.cuh"

__global__ void vorticity(float* vField, Particle* field, size_t xSize, size_t ySize);
__device__ float curl(Particle* curField, size_t xSize, size_t ySize, int x, int y);
__global__ void applyVorticity(Particle* newField, Particle* curField, float* vField, size_t xSize, size_t ySize, float vorticity, float dt);
__device__ Vec2f absGradient(float* vorticityField, size_t xSize, size_t ySize, int x, int y);

void diffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt);
__global__ void computeColor(Particle* newField, Particle* curField, size_t xSize, size_t ySize, float cDiffusion, float dt);
__device__ Color3f jacobiColor(Particle* colorField, size_t xSize, size_t ySize, int x, int y, Color3f c, float alpha, float beta);
__global__ void diffuse(Particle* newField, Particle* curField, size_t xSize, size_t ySize, float vDiffusion, float dt);
__device__ Vec2f jacobiVelocity(Particle* field, size_t xSize, size_t ySize, int x, int y, Vec2f flowVel, float alpha, float beta);

__global__ void applyForceandColor(Particle* curField, size_t xSize, size_t ySize, Color3f curColor, Vec2f force, Vec2f pos, int radius, float dt);

void pressure(dim3 numBlocks, dim3 threadsPerBlock, float dt);
__global__ void applyPressure(Particle* field, size_t xSize, size_t ySize, float* pNew, float* pOld, float pressure, float dt);
__device__ float divergence(Particle* field, size_t xSize, size_t ySize, int x, int y);
__device__ float jacobiPressure(float* pressureField, size_t xSize, size_t ySize, int x, int y, float B, float alpha, float beta);

__global__ void projection(Particle* newField, size_t xSize, size_t ySize, float* pField);
__device__ Vec2f gradient(float* field, size_t xSize, size_t ySize, int x, int y);

__global__ void advection(Particle* newField, Particle* curField, size_t xSize, size_t ySize, float dDiffusion, float dt);
__device__ Particle interpolate(Vec2f v, Particle* field, size_t xSize, size_t ySize);

__global__ void draw(unsigned char* colorField, Particle* field, size_t xSize, size_t ySize);


// Allocate buffers
void cudaInit(size_t cudaX, size_t cudaY) {
	xSize = cudaX, ySize = cudaY;

	cudaMalloc(&curField,       xSize*ySize * sizeof(Particle));
	cudaMalloc(&newField,       xSize*ySize * sizeof(Particle));

	cudaMalloc(&colorField,     xSize*ySize * sizeof(float));
	cudaMalloc(&colorField,     xSize*ySize * sizeof(float));
	
	cudaMalloc(&curPressure,    xSize*ySize * sizeof(float));
	cudaMalloc(&newPressure,    xSize*ySize * sizeof(float));
	
	cudaMalloc(&vorticityField, xSize*ySize * sizeof(float));

	// 1: Reds, 2: Blues, 3: Yellows, 4: Greens
	int colorSetting = 2;
	curColor = colorArray[0];

	switch (colorSetting) {
		case 1: {
			// Fire, Hot!
			colorArray[0] = { 0.76f, 0.24f, 0.019f };
			colorArray[1] = { 0.80f, 0.27f, 0.06f };
			colorArray[2] = { 0.847f, 0.309f, 0.098f };
			colorArray[3] = { 0.886f, 0.345f, 0.133f };
			colorArray[4] = { 0.925f, 0.38f, 0.164f };
			colorArray[5] = { 0.968f, 0.415f, 0.196f };
			colorArray[6] = { 1.0f, 0.451f, 0.231f };
		}//case 1
		break;

		case 2: {
			// Ice, Ice... Baby?
			colorArray[0] = { 0.019f, 0.24f, 0.76f };
			colorArray[1] = { 0.06f, 0.27f, 0.80f };
			colorArray[2] = { 0.098f, 0.309f, 0.847f };
			colorArray[3] = { 0.133f, 0.345f, 0.886f };
			colorArray[4] = { 0.164f, 0.38f, 0.925f };
			colorArray[5] = { 0.196f, 0.415f, 0.968f };
			colorArray[6] = { 0.231f, 0.451f, 1.0f };
		}//case 2
		break;

		case 3: {
			// Holy-moly
			colorArray[0] = { 1.0f, 1.0f, 0.471f };
			colorArray[1] = { 0.922f, 0.843f, 0.471f };
			colorArray[2] = { 0.922f, 0.922f, 0.706f };
			colorArray[3] = { 0.961f, 0.922f, 0.588f };
			colorArray[4] = { 1.0f, 1.0f, 0.471f };
			colorArray[5] = { 1.0f, 0.529f, 0.0f };
			colorArray[6] = { 0.608f, 0.608f, 0.49f };
		}//case 3
		break;

		case 4: {
			// Acid
			colorArray[0] = { 0.153f, 0.749f, 0.098f };
			colorArray[1] = { 0.263f, 0.694f, 0.196f };
			colorArray[2] = { 0.325f, 0.643f, 0.258f };
			colorArray[3] = { 0.364f, 0.588f, 0.305f };
			colorArray[4] = { 0.392f, 0.533f, 0.349f };
			colorArray[5] = { 0.412f, 0.478f, 0.388f };
			colorArray[6] = { 0.423f, 0.423f, 0.423f };
		}
		break;

		default: {
			// Fire, Hot!
			colorArray[0] = { 0.76f, 0.24f, 0.019f };
			colorArray[1] = { 0.80f, 0.27f, 0.06f };
			colorArray[2] = { 0.847f, 0.309f, 0.098f };
			colorArray[3] = { 0.886f, 0.345f, 0.133f };
			colorArray[4] = { 0.925f, 0.38f, 0.164f };
			colorArray[5] = { 0.968f, 0.415f, 0.196f };
			colorArray[6] = { 1.0f, 0.451f, 0.231f };
		}//default
		break;
	}//switch
}//cudaInit()

// Computes vorticity -> diffusion -> force -> pressure -> projection -> advection -> draw
void compute(unsigned char* result, int x1pos, int y1pos, int x2pos, int y2pos, float dt, bool state) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	dim3 threadsPerBlock(gpuData.xThreads, gpuData.yThreads);
	int xBlock = (unsigned int)xSize / threadsPerBlock.x;
	int yBlock = (unsigned int)ySize / threadsPerBlock.y;
	dim3 numBlocks(xBlock, yBlock);

	cudaDeviceSynchronize();

	// Compute and apply the curl of the flow velocity
	vorticity<<<numBlocks, threadsPerBlock>>>(vorticityField, curField, xSize, ySize);
	applyVorticity<<<numBlocks, threadsPerBlock>>>(newField, curField, vorticityField, xSize, ySize, data.vorticity, dt);
	swap(curField, newField);

	// Diffuse velocity and color
	diffusion(numBlocks, threadsPerBlock, dt);

	// Apply color and force
	if (state) {
		elapsedTime += dt;
		// Apply a gradient, cycling through the colorArray
		curColor = colorArray[int(elapsedTime) % colorSize] + colorArray[int((elapsedTime)+1) % colorSize];

		Vec2f force;
		force.x = (x2pos - x1pos)*data.forceScale;
		force.y = (y2pos - y1pos)*data.forceScale;
		Vec2f pos = { float(x2pos), float(y2pos) }; // Cursor position
		applyForceandColor<<<numBlocks, threadsPerBlock>>> (curField, xSize, ySize, curColor, force, pos, data.radius, dt);
	}//if

	// Compute pressure
	pressure(numBlocks, threadsPerBlock, dt);

	// Compute projection
	projection<<<numBlocks, threadsPerBlock>>> (curField, xSize, ySize, curPressure);
	cudaMemset(curPressure, 0, xSize*ySize*sizeof(float)); // Reset pressure

	// Compute advection
	advection<<<numBlocks, threadsPerBlock>>> (newField, curField, xSize, ySize, data.densityDiffusion, dt);
	swap(newField, curField);

	// Draw the image
	draw<<<numBlocks, threadsPerBlock>>> (colorField, curField, xSize, ySize);
	cudaEventRecord(end);

	// Copy image to cpu
	cudaMemcpy(result, colorField, xSize*ySize*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(end);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, end);
	//printf("Time for compute(): %f ms\n", ms);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		cout << cudaGetErrorName(err) << endl;
		//printf("xSize: %d\nySize: %d\n", xSize, ySize);
		//printf("threadsPerBlock: { %d*%d = %d }  |  Blocks: { %d*%d = %d }\n\n",
		//gpuData.xThreads,
		//gpuData.yThreads,
		//gpuData.xThreads*gpuData.yThreads,
		//xBlock,
		//yBlock,
		//xBlock * yBlock);
		//printf("%d / %d = %d\n", (unsigned int)xSize, threadsPerBlock.x, xBlock);
		//printf("%d / %d = %d\n", (unsigned int)ySize, threadsPerBlock.y, yBlock);
	}//if
}//compute()

// Computes vorticity - the curl of flow velocity
__global__ void vorticity(float* vorticityField, Particle* curField, size_t xSize, size_t ySize) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	vorticityField[x + y*xSize] = curl(curField, xSize, ySize, x, y);
}//vorticity()

// Computes curl behavior of a vector flow field
__device__ float curl(Particle* curField, size_t xSize, size_t ySize, int x, int y) {
	Vec2f curl = curField[x + y*xSize].vel;
	float x1 = -curl.x, x2 = -curl.x, y1 = -curl.y, y2 = -curl.y;

	// Check bounds for velocity assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = curField[int(x+1) + int(y) * xSize].vel.x;

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = curField[int(x-1) + int(y) * xSize].vel.x;

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = curField[int(x) + int(y+1) * xSize].vel.y;

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y - 1) < ySize)
		y2 = curField[int(x) + int(y-1) * xSize].vel.y;

	return ((y1 - y2) - (x1 - x2))/2;
}//curl()

// Applies vorticity to the particle field
__global__ void applyVorticity(Particle *newField, Particle *curField, float *vorticityField, size_t xSize, size_t ySize, float vorticity, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Particle &curParticle = curField[x + y * xSize];
	Particle &newParticle = newField[x + y * xSize];

	// Compute absolute gradient vector
	Vec2f vGradient = absGradient(vorticityField, xSize, ySize, x, y);
	vGradient.y *= -1.0f; // Change sign
	
	// Square the sums of the squared cooridinates (with a slight offset)
	float length = sqrtf(
		((vGradient.x * vGradient.x) + 1e-10f) + ((vGradient.y * vGradient.y) + 1e-10f)
	);
	Vec2f vNorm = vGradient * (1.0f / length);

	Vec2f vForce = vNorm * vorticityField[x + y*xSize] * vorticity; // Force of the vector
	newParticle = curParticle; // Update new particles
	newParticle.vel = newParticle.vel + vForce * dt; // velocity + impulse = new particle velocity
}//applyVorticity()

// Computes the absolute value gradient of a vorticity field
__device__ Vec2f absGradient(float *vorticityField, size_t xSize, size_t ySize, int x, int y) {
	float curl = vorticityField[x + y*xSize];
	float x1 = curl, x2 = curl, y1 = curl, y2 = curl;

	// Check bounds for vorticity assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = vorticityField[int(x+1) + int(y) * xSize];

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = vorticityField[int(x-1) + int(y) * xSize];

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = vorticityField[int(x) + int(y+1) * xSize];

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = vorticityField[int(x) + int(y-1) * xSize];

	return { (abs(x1) - abs(x2))/2, (abs(y1) - abs(y2))/2 };
}//absGradient()

// Diffuses velocity and color fields
void diffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt) {
	for (int i = 0; i < gpuData.velocitySize; ++i) {
		diffuse<<<numBlocks, threadsPerBlock>>>(newField, curField, xSize, ySize, data.velocityDiffusion, dt);
		computeColor<<<numBlocks, threadsPerBlock>>>(newField, curField, xSize, ySize, data.colorDiffusion, dt);
		swap(newField, curField);
	}//for
}//diffusion()

// Computes diffusion divergency of velocity field
__global__ void diffuse(Particle* newField, Particle* curField, size_t xSize, size_t ySize, float velocityDiffusion, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Vec2f flowVel = curField[x + y*xSize].vel;

	float alpha = velocityDiffusion*velocityDiffusion / dt;
	float beta = 4.0f + alpha;
	newField[x + y*xSize].vel = jacobiVelocity(curField, xSize, ySize, x, y, flowVel, alpha, beta);
}//diffuse()

// Computes a Jacobi iteration on velocity grid field
__device__ Vec2f jacobiVelocity(Particle* field, size_t xSize, size_t ySize, int x, int y, Vec2f flowVel, float alpha, float beta) {
	Vec2f x1 = flowVel, x2 = flowVel, y1 = flowVel, y2 = flowVel;

	// Check bounds for jacobi velocity assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = field[int(x+1) + int(y) * xSize].vel;

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = field[int(x-1) + int(y) * xSize].vel;

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = field[int(x) + int(y+1) * xSize].vel;

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = field[int(x) + int(y-1) * xSize].vel;

	return (x1 + x2 + y1 + y2 + flowVel * alpha) / beta;
}//jacobiVelocity()

// Computes color field diffusion
__global__ void computeColor(Particle* newField, Particle* curField, size_t xSize, size_t ySize, float colorDiffusion, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Color3f c = curField[x + y*xSize].color;

	float alpha = colorDiffusion*colorDiffusion / dt;
	float beta = 4.0f + alpha;
	newField[x + y*xSize].color = jacobiColor(curField, xSize, ySize, x, y, c, alpha, beta);
}//computeColor()

// Computes a Jacobi iteration on color grid field
__device__ Color3f jacobiColor(Particle* colorField, size_t xSize, size_t ySize, int x, int y, Color3f c, float alpha, float beta) {
	Color3f x1, x2, y1, y2;

	// Check bounds for jacobi color grid assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = colorField[int(x+1) + int(y) * xSize].color;

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = colorField[int(x-1) + int(y) * xSize].color;

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = colorField[int(x) + int(y+1) * xSize].color;

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = colorField[int(x) + int(y-1) * xSize].color;

	return { (x1 + x2 + y1 + y2 + c * alpha) * (1.0f / beta) };
}//jacobiColor

// Computes force and add color dye to the particle field
__global__ void applyForceandColor(Particle* curField, size_t xSize, size_t ySize, Color3f curColor, Vec2f force, Vec2f pos, int radius, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float impulsePos = expf(-((x - pos.x)*(x - pos.x) + (y - pos.y)*(y - pos.y)) / radius);
	
	Particle &particle = curField[x + y*xSize];
	particle.vel = particle.vel + force*dt*impulsePos;
	
	curColor = curColor*impulsePos + particle.color;
	particle.color.R = curColor.R;
	particle.color.G = curColor.G;
	particle.color.B = curColor.B;
}//applyForceandColor()

// performs several iterations over pressure field
void pressure(dim3 numBlocks, dim3 threadsPerBlock, float dt) {
	for (int i = 0; i < gpuData.pressureSize; ++i){
		applyPressure<<<numBlocks, threadsPerBlock>>> (curField, xSize, ySize, newPressure, curPressure, data.pressure, dt);
		swap(curPressure, newPressure);
	}//for
}//pressure()

// Applies pressure using divergence and a Jacobi iteration
__global__ void applyPressure(Particle* curField, size_t xSize, size_t ySize, float *newPressure, float* curPressure, float pressure, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float d = divergence(curField, xSize, ySize, x, y);

	float alpha = -1.0f * pressure*pressure;
	float beta = 4.0f;
	newPressure[x + y*xSize] = jacobiPressure(curPressure, xSize, ySize, x, y, d, alpha, beta);
}//applyPressure()

// Computes divergency of a velocity field
__device__ float divergence(Particle* curField, size_t xSize, size_t ySize, int x, int y){
	Particle& divField = curField[x + y*xSize];
	float x1 = -1.0f*divField.vel.x, x2 = -1.0f*divField.vel.x, y1 = -1.0f*divField.vel.y, y2 = -1.0f*divField.vel.y;

	// Check bounds for divergency of velocity assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = curField[int(x+1) + int(y) * xSize].vel.x;

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = curField[int(x-1) + int(y) * xSize].vel.x;

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = curField[int(x) + int(y+1) * xSize].vel.y;

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = curField[int(x) + int(y-1) * xSize].vel.y;

	return (x1-x2 + y1-y2)/2;
}//divergence()

// Computes a Jacobi iteration of a pressure grid field
__device__ float jacobiPressure(float* pressureField, size_t xSize, size_t ySize, int x, int y, float d, float alpha, float beta) {
	float pressure = pressureField[x + y*xSize];
	float x1 = pressure, x2 = pressure, y1 = pressure, y2 = pressure;

	// Check bounds for jacobi pressure assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = pressureField[int(x+1) + int(y) * xSize];

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = pressureField[int(x-1) + int(y) * xSize];

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = pressureField[int(x) + int(y+1) * xSize];

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = pressureField[int(x) + int(y-1) * xSize];

	return (x1 + x2 + y1 + y2 + alpha * d) / beta;
}//jacobiPressure()

// Computes projection pressure field for a velocity field
__global__ void projection(Particle *curField, size_t xSize, size_t ySize, float *curPressure){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Vec2f &projectionField = curField[x + y*xSize].vel;
	projectionField = projectionField - gradient(curPressure, xSize, ySize, x, y);
}//projection()

// Computes a gradient of the pressure field
__device__ Vec2f gradient(float* curPressure, size_t xSize, size_t ySize, int x, int y) {
	float pressure = curPressure[x + y*xSize];
	float x1 = pressure, x2 = pressure, y1 = pressure, y2 = pressure;

	// Check bounds for pressure assignment
	if ((x+1) >= 0 && y >= 0 && (x+1) < xSize && y < ySize)
		x1 = curPressure[int(x+1) + int(y) * xSize];

	if ((x-1) >= 0 && y >= 0 && (x-1) < xSize && y < ySize)
		x2 = curPressure[int(x-1) + int(y) * xSize];

	if (x >= 0 && (y+1) >= 0 && x < xSize && (y+1) < ySize)
		y1 = curPressure[int(x) + int(y+1) * xSize];

	if (x >= 0 && (y-1) >= 0 && x < xSize && (y-1) < ySize)
		y2 = curPressure[int(x) + int(y-1) * xSize];

	return { (x1 - x2)/2, (y1 - y2)/2 };
}//gradient()

// Computes bulk motion of particles using interpolation
__global__ void advection(Particle *newField, Particle *curField, size_t xSize, size_t ySize, float densityDiffusion, float dt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float decay = 1.0f / (1.0f + densityDiffusion*dt);
	Vec2f pos = { float(x), float(y) };

	Particle &curPressure = curField[x + y*xSize];	
	Particle pInterp = interpolate(pos - curPressure.vel*dt, curField, xSize, ySize);
	pInterp.vel = pInterp.vel*decay;

	pInterp.color.R = min(1.0f, pow(pInterp.color.R, 1.0125f) * decay);
	pInterp.color.G = min(1.0f, pow(pInterp.color.G, 1.0125f) * decay);
	pInterp.color.B = min(1.0f, pow(pInterp.color.B, 1.0125f) * decay);
	newField[x + y*xSize] = pInterp;
}//advection()

// Interpolates the quantity of cells
__device__ Particle interpolate(Vec2f viscosity, Particle *curField, size_t xSize, size_t ySize) {
	float x1 = (int)viscosity.x, x2 = (int)viscosity.x+1, y1 = (int)viscosity.y, y2 = (int)viscosity.y+1;
	Particle px1, px2, px3, px4;

	// Check bounds for the min of max viscosity
	px1 = curField[int(min(xSize - 1.0f, max(0.0f, x1))) + int(min(ySize - 1.0f, max(0.0f, y1)))*xSize];
	px2 = curField[int(min(xSize - 1.0f, max(0.0f, x1))) + int(min(ySize - 1.0f, max(0.0f, y2)))*xSize];
	px3 = curField[int(min(xSize - 1.0f, max(0.0f, x2))) + int(min(ySize - 1.0f, max(0.0f, y1)))*xSize];
	px4 = curField[int(min(xSize - 1.0f, max(0.0f, x2))) + int(min(ySize - 1.0f, max(0.0f, y2)))*xSize];

	float x2Viscosity = (x2 - viscosity.x) / (x2 - x1);
	float x1Viscosity = (viscosity.x - x1) / (x2 - x1);
	float y2Viscosity = (y2 - viscosity.y) / (y2 - y1);
	float y1Viscosity = (viscosity.y - y1) / (y2 - y1);

	Vec2f oddFriction = px1.vel*x2Viscosity + px3.vel*x1Viscosity;
	Vec2f evenFriction = px2.vel*x2Viscosity + px4.vel*x1Viscosity;
	
	Color3f color = px2.color*x2Viscosity + px4.color*x1Viscosity;
	
	Particle result;
	result.vel = oddFriction * y2Viscosity + evenFriction * y1Viscosity;
	result.color = color * y2Viscosity + color * y1Viscosity;
	return result;
}//interpolate()

// Draws to the colorField
__global__ void draw(unsigned char *colorField, Particle *curField, size_t xSize, size_t ySize) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float R = curField[x + y*xSize].color.R;
	float G = curField[x + y*xSize].color.G;
	float B = curField[x + y*xSize].color.B;

	colorField[4*(x + y*xSize) + 0] = min(255.0f, 255.0f*R);
	colorField[4*(x + y*xSize) + 1] = min(255.0f, 255.0f*G);
	colorField[4*(x + y*xSize) + 2] = min(255.0f, 255.0f*B);
	colorField[4*(x + y*xSize) + 3] = 255.0f;
}//draw()

// Releases all allocated resources
void cudaExit() {
	cudaFree(curField);
	cudaFree(newField);
	
	cudaFree(colorField);
	
	cudaFree(curPressure);
	cudaFree(newPressure);
	
	cudaFree(vorticityField);
}//cudaExit()