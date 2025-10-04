#include "pathtrace.h"


#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "optixDenoiser.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

struct is_terminated {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces <= 0;
    }
};

__global__ void convertFloat3ToFloat4Kernel(glm::vec3* src, float4* dst, unsigned int numPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    glm::vec3 pixel = src[idx];
    dst[idx] = make_float4(pixel.x, pixel.y, pixel.z, 1.0f);
}

__global__ void convertFloat4ToFloat3Kernel(float4* src, glm::vec3* dst, unsigned int numPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    float4 pixel = src[idx];
    dst[idx] = glm::vec3(pixel.x, pixel.y, pixel.z);
}

extern "C" void launchConvertFloat3ToFloat4(
    glm::vec3* src, float4* dst, unsigned int numPixels)
{
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    convertFloat3ToFloat4Kernel << <numBlocks, blockSize >> > (src, dst, numPixels);
    cudaDeviceSynchronize();
}

extern "C" void launchConvertFloat4ToFloat3(
    float4* src, glm::vec3* dst, unsigned int numPixels)
{
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    convertFloat4ToFloat3Kernel << <numBlocks, blockSize >> > (src, dst, numPixels);
    cudaDeviceSynchronize();
}

__global__ void captureNormalsAndAlbedo(
    int num_paths,
    PathSegment* paths,
    ShadeableIntersection* intersections,
    glm::vec3* normals,
    glm::vec3* albedo,
    Material* materials,
    int iter,
    int width,
    int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& path = paths[idx];
    ShadeableIntersection& isect = intersections[idx];

    int pixelIdx = path.pixelIndex;
    if (pixelIdx < 0 || pixelIdx >= width * height) return;

    // Only capture on first bounce for clean normals/albedo
    if (isect.t > 0.0f) {
        // Capture normal (world space, normalized)
        if (normals) {
            normals[pixelIdx] = glm::normalize(isect.surfaceNormal);
        }

        // Capture albedo (material base color)
        if (albedo && isect.materialId >= 0) {
            albedo[pixelIdx] = materials[isect.materialId].color;
        }
    }
    else {
        // No intersection - set to background
        if (normals) {
            normals[pixelIdx] = glm::vec3(0.0f);
        }
        if (albedo) {
            albedo[pixelIdx] = glm::vec3(0.0f);
        }
    }
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static EnvironmentMap dev_environmentMap;
static glm::vec3* dev_envmap_data = NULL;
static LightInfo* dev_lights = NULL;
static int num_lights = 0;
static Triangle* dev_triangles = NULL;
static int total_triangles = 0;
static GPUTexture* dev_textures = NULL;
static int num_textures = 0;
static int* dev_material_ids = NULL;  // For sorting keys
static int* dev_path_indices = NULL;  // For sorting values
static PathSegment* dev_paths_sorted = NULL;
static ShadeableIntersection* dev_intersections_sorted = NULL;
static std::vector<BVH*> dev_bvhs;  // One BVH per mesh
static BVHNode** dev_bvh_nodes = nullptr;
static int** dev_bvh_triangle_indices = nullptr;
static int num_meshes_with_bvh = 0;
static bool USE_BVH = true;  // Toggle BVH usage
static int BVH_MAX_TREE_DEPTH = 24;  // Configurable max depth

OptiXDenoiser* g_denoiser = nullptr;

// Device buffers for denoising
glm::vec3* dev_normals = NULL;     // Normal buffer for denoising
glm::vec3* dev_albedo = NULL;      // Albedo buffer for denoising (optional)
glm::vec3* dev_denoised = NULL;    // Denoised output buffer

// Denoiser control flags
bool USE_DENOISER = true;          // Enable/disable denoiser
bool DENOISE_WITH_NORMALS = true;  // Use normal buffer as guide
bool DENOISE_WITH_ALBEDO = false;  // Use albedo buffer as guide (optional)
int DENOISE_START_ITER = 1;        // Start denoising after this many iterations
int DENOISE_FREQUENCY = 1;         // Apply denoiser every N iterations



void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Get the area of a geometry
__host__ __device__ float getGeomArea(const Geom& geom) {
    if (geom.type == SPHERE) {
        float radius = geom.scale.x * 0.5f;
        return 4.0f * PI * radius * radius;
    }
    else if (geom.type == CUBE) {
        float sx = geom.scale.x;
        float sy = geom.scale.y;
        float sz = geom.scale.z;
        return 2.0f * (sx * sy + sy * sz + sz * sx);
    }
    return 1.0f; // Default
}

void initializeLights(Scene* scene) {
    // Count emissive objects
    std::vector<LightInfo> lightInfos;
    float totalArea = 0.0f;

    for (int i = 0; i < scene->geoms.size(); i++) {
        int matId = scene->geoms[i].materialid;
        //if (matId < 0) continue;
        if (scene->materials[matId].emittance > 0.0f) {
            LightInfo info;
            info.geomIdx = i;
            info.area = getGeomArea(scene->geoms[i]);
            totalArea += info.area;
            lightInfos.push_back(info);
        }
    }

    // Normalize PDFs
    for (auto& light : lightInfos) {
        light.pdf = light.area / totalArea;
    }

    num_lights = lightInfos.size();
    if (num_lights > 0) {
        cudaMalloc(&dev_lights, num_lights * sizeof(LightInfo));
        cudaMemcpy(dev_lights, lightInfos.data(), num_lights * sizeof(LightInfo),
            cudaMemcpyHostToDevice);
    }

    //printf("Initialized %d light sources for MIS\n", num_lights);
}

void initializeBVHs(Scene* scene) {
    // Count meshes
    num_meshes_with_bvh = 0;
    for (const auto& geom : scene->geoms) {
        if (geom.type == GLTF_MESH && geom.meshData != nullptr) {
            num_meshes_with_bvh++;
        }
    }

    if (num_meshes_with_bvh == 0 || !USE_BVH) return;

    // Clear existing BVHs
    for (auto* bvh : dev_bvhs) {
        delete bvh;
    }
    dev_bvhs.clear();

    // Allocate arrays for device pointers
    BVHNode** host_bvh_nodes = new BVHNode * [num_meshes_with_bvh];
    int** host_bvh_indices = new int* [num_meshes_with_bvh];

    int meshIdx = 0;
    for (int g = 0; g < scene->geoms.size(); g++) {
        if (scene->geoms[g].type == GLTF_MESH && scene->geoms[g].meshData != nullptr) {
            // Build BVH for this mesh
            BVH* bvh = new BVH();
            bvh->build(scene->geoms[g].meshData->triangles, BVH_MAX_TREE_DEPTH);
            bvh->uploadToGPU();

            // Store BVH and get device pointers
            dev_bvhs.push_back(bvh);
            host_bvh_nodes[meshIdx] = bvh->getDeviceNodes();
            host_bvh_indices[meshIdx] = bvh->getDeviceTriangleIndices();

            // Print stats for this mesh
            std::cout << "Mesh " << g << " BVH stats:" << std::endl;
            bvh->getStats().print();

            // Store BVH index in geom (you'll need to add a bvhIndex field to Geom struct)
            scene->geoms[g].bvhIndex = meshIdx;

            meshIdx++;
        }
        else {
            scene->geoms[g].bvhIndex = -1;  // No BVH for non-mesh objects
        }
    }

    // Upload pointer arrays to GPU
    cudaMalloc(&dev_bvh_nodes, num_meshes_with_bvh * sizeof(BVHNode*));
    cudaMalloc(&dev_bvh_triangle_indices, num_meshes_with_bvh * sizeof(int*));

    cudaMemcpy(dev_bvh_nodes, host_bvh_nodes,
        num_meshes_with_bvh * sizeof(BVHNode*), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bvh_triangle_indices, host_bvh_indices,
        num_meshes_with_bvh * sizeof(int*), cudaMemcpyHostToDevice);

    delete[] host_bvh_nodes;
    delete[] host_bvh_indices;

    // Update geoms on device
    cudaMemcpy(dev_geoms, scene->geoms.data(),
        scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    std::cout << "Initialized BVH for " << num_meshes_with_bvh << " meshes" << std::endl;
}

__host__ void buildEnvMapDistribution(
    const HostEnvironmentMap& hostEnvMap,  // CPU-side environment map
    EnvMapDistribution& distribution)       // Distribution to build
{
    if (!hostEnvMap.enabled || hostEnvMap.data.empty()) {
        distribution.marginalCDF = nullptr;
        distribution.conditionalCDFs = nullptr;
        distribution.totalPower = 0.0f;
        return;
    }

    distribution.width = hostEnvMap.width;
    distribution.height = hostEnvMap.height;

    // Allocate host memory for computation
    float* luminances = new float[hostEnvMap.width * hostEnvMap.height];
    float* rowPowers = new float[hostEnvMap.height];

    // Compute luminance for each pixel
    for (int y = 0; y < hostEnvMap.height; y++) {
        for (int x = 0; x < hostEnvMap.width; x++) {
            int idx = y * hostEnvMap.width + x;
            glm::vec3 color = hostEnvMap.data[idx];

            // Use standard luminance formula
            luminances[idx] = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;

            // Weight by sin(theta) for equirectangular projection
            float theta = (y + 0.5f) * PI / hostEnvMap.height;
            luminances[idx] *= sinf(theta);
        }
    }

    // Build conditional CDFs (for each row)
    float* hostConditionalCDFs = new float[hostEnvMap.width * hostEnvMap.height];

    for (int y = 0; y < hostEnvMap.height; y++) {
        float rowSum = 0.0f;
        for (int x = 0; x < hostEnvMap.width; x++) {
            int idx = y * hostEnvMap.width + x;
            rowSum += luminances[idx];
            hostConditionalCDFs[idx] = rowSum;
        }

        // Normalize the row CDF
        rowPowers[y] = rowSum;
        if (rowSum > 0) {
            float invRowSum = 1.0f / rowSum;
            for (int x = 0; x < hostEnvMap.width; x++) {
                hostConditionalCDFs[y * hostEnvMap.width + x] *= invRowSum;
            }
        }
    }

    // Build marginal CDF (for selecting rows)
    float* hostMarginalCDF = new float[hostEnvMap.height];

    float totalSum = 0.0f;
    for (int y = 0; y < hostEnvMap.height; y++) {
        totalSum += rowPowers[y];
        hostMarginalCDF[y] = totalSum;
    }

    distribution.totalPower = totalSum;

    // Normalize marginal CDF
    if (totalSum > 0) {
        float invTotalSum = 1.0f / totalSum;
        for (int y = 0; y < hostEnvMap.height; y++) {
            hostMarginalCDF[y] *= invTotalSum;
        }
    }

    // Allocate GPU memory and copy
    cudaMalloc(&distribution.conditionalCDFs,
        hostEnvMap.width * hostEnvMap.height * sizeof(float));
    cudaMemcpy(distribution.conditionalCDFs, hostConditionalCDFs,
        hostEnvMap.width * hostEnvMap.height * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMalloc(&distribution.marginalCDF, hostEnvMap.height * sizeof(float));
    cudaMemcpy(distribution.marginalCDF, hostMarginalCDF,
        hostEnvMap.height * sizeof(float), cudaMemcpyHostToDevice);

    // Cleanup host memory
    delete[] luminances;
    delete[] rowPowers;
    delete[] hostConditionalCDFs;
    delete[] hostMarginalCDF;
}


void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	// Initialize lights for MIS
	initializeLights(scene);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    dev_environmentMap.enabled = scene->environmentMap.enabled;
    dev_environmentMap.width = scene->environmentMap.width;
    dev_environmentMap.height = scene->environmentMap.height;
    dev_environmentMap.intensity = scene->environmentMap.intensity;

    if (scene->environmentMap.enabled && scene->environmentMap.data.size() > 0)
    {
        size_t envMapSize = scene->environmentMap.width * scene->environmentMap.height * sizeof(glm::vec3);
        cudaMalloc(&dev_envmap_data, envMapSize);
        cudaMemcpy(dev_envmap_data, scene->environmentMap.data.data(), envMapSize, cudaMemcpyHostToDevice);
        dev_environmentMap.data = dev_envmap_data;
        buildEnvMapDistribution(scene->environmentMap, dev_environmentMap.distribution);

        //printf("Environment map uploaded to GPU: %dx%d pixels, %.2f MB\n",
        //    scene->environmentMap.width, scene->environmentMap.height,
        //    envMapSize / (1024.0f * 1024.0f));
    }
    else
    {
        dev_environmentMap.data = nullptr;
        dev_environmentMap.distribution.marginalCDF = nullptr;
        dev_environmentMap.distribution.conditionalCDFs = nullptr;
        dev_environmentMap.distribution.totalPower = 0.0f;
    }

    total_triangles = 0;
    for (const auto& mesh : scene->meshes) {
        total_triangles += mesh.triangles.size();
    }

    if (total_triangles > 0) {
        cudaMalloc(&dev_triangles, total_triangles * sizeof(Triangle));

        // Copy triangles to GPU
        int offset = 0;
        for (int g = 0; g < scene->geoms.size(); g++) {
            if (scene->geoms[g].type == GLTF_MESH && scene->geoms[g].meshData != nullptr) {
                // Set triangle start index for this mesh
                hst_scene->geoms[g].triangleStart = offset;

                // Copy triangles
                cudaMemcpy(dev_triangles + offset,
                    scene->geoms[g].meshData->triangles.data(),
                    scene->geoms[g].meshData->triangles.size() * sizeof(Triangle),
                    cudaMemcpyHostToDevice);

                offset += scene->geoms[g].meshData->triangles.size();
            }
        }

        // Copy updated geoms with triangle indices
        cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    }

    // Upload textures to GPU
    num_textures = scene->textures.size();
    if (num_textures > 0) {
        cudaMalloc(&dev_textures, num_textures * sizeof(GPUTexture));

        // Create temporary array to hold GPU texture info
        std::vector<GPUTexture> gpuTextures(num_textures);

        // Upload each texture
        for (int i = 0; i < num_textures; i++) {
            const Texture& cpuTex = scene->textures[i];
            GPUTexture& gpuTex = gpuTextures[i];

            gpuTex.width = cpuTex.width;
            gpuTex.height = cpuTex.height;
            gpuTex.components = cpuTex.components;

            // Allocate GPU memory for texture data
            size_t texSize = cpuTex.width * cpuTex.height * cpuTex.components;
            cudaMalloc(&gpuTex.data, texSize);
            cudaMemcpy(gpuTex.data, cpuTex.data, texSize, cudaMemcpyHostToDevice);

            //printf("Uploaded texture %d: %dx%d, %d channels, %.2f MB\n",
            //    i, cpuTex.width, cpuTex.height, cpuTex.components,
            //    texSize / (1024.0f * 1024.0f));
        }

        // Copy the array of texture descriptors to GPU
        cudaMemcpy(dev_textures, gpuTextures.data(),
            num_textures * sizeof(GPUTexture), cudaMemcpyHostToDevice);
    }

    if (total_triangles > 0) {
        initializeBVHs(scene);
    }

    // Allocate sorting buffers if material sorting is enabled
#if MATERIAL_SORTING
    cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));
    cudaMalloc(&dev_path_indices, pixelcount * sizeof(int));
    checkCUDAError("pathtraceInit - sorting buffers");
    cudaMalloc(&dev_paths_sorted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_sorted, pixelcount * sizeof(ShadeableIntersection));
    checkCUDAError("pathtraceInit - sorted data buffers");
#endif

    if (USE_DENOISER) {
        cudaMalloc(&dev_normals, pixelcount * sizeof(glm::vec3));
        cudaMemset(dev_normals, 0, pixelcount * sizeof(glm::vec3));

        if (DENOISE_WITH_ALBEDO) {
            cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
            cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
        }

        cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
        cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

        // Initialize OptiX denoiser
        if (!g_denoiser) {
            g_denoiser = new OptiXDenoiser();
            g_denoiser->init(cam.resolution.x, cam.resolution.y,
                DENOISE_WITH_NORMALS, DENOISE_WITH_ALBEDO);
            //std::cout << "OptiX Denoiser initialized" << std::endl;
        }
    }
    

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    // Free environment map data
    if (dev_envmap_data != NULL)
    {
        cudaFree(dev_envmap_data);
        dev_envmap_data = NULL;
    }

    if (dev_environmentMap.distribution.marginalCDF) {
        cudaFree(dev_environmentMap.distribution.marginalCDF);
    }
    if (dev_environmentMap.distribution.conditionalCDFs) {
        cudaFree(dev_environmentMap.distribution.conditionalCDFs);
    }

    if (dev_lights != NULL) {
        cudaFree(dev_lights);
        dev_lights = NULL;
    }

    if (dev_triangles != NULL) {
        cudaFree(dev_triangles);
        dev_triangles = NULL;
    }

    if (dev_textures != NULL && num_textures > 0) {
        // First, get the texture descriptors to free individual texture data
        std::vector<GPUTexture> gpuTextures(num_textures);
        cudaMemcpy(gpuTextures.data(), dev_textures,
            num_textures * sizeof(GPUTexture), cudaMemcpyDeviceToHost);

        // Free each texture's data
        for (int i = 0; i < num_textures; i++) {
            if (gpuTextures[i].data != nullptr) {
                cudaFree(gpuTextures[i].data);
            }
        }

        // Free the texture descriptor array
        cudaFree(dev_textures);
        dev_textures = NULL;
        num_textures = 0;
    }

#if MATERIAL_SORTING
    if (dev_material_ids) {
        cudaFree(dev_material_ids);
        dev_material_ids = NULL;
    }
    if (dev_path_indices) {
        cudaFree(dev_path_indices);
        dev_path_indices = NULL;
    }
    if (dev_paths_sorted) {
        cudaFree(dev_paths_sorted);
        dev_paths_sorted = NULL;
    }
    if (dev_intersections_sorted) {
        cudaFree(dev_intersections_sorted);
        dev_intersections_sorted = NULL;
    }
#endif

    for (auto* bvh : dev_bvhs) {
        delete bvh;
    }
    dev_bvhs.clear();

    if (dev_bvh_nodes) {
        cudaFree(dev_bvh_nodes);
        dev_bvh_nodes = nullptr;
    }

    if (dev_bvh_triangle_indices) {
        cudaFree(dev_bvh_triangle_indices);
        dev_bvh_triangle_indices = nullptr;
    }

    // Free denoising buffers
    if (dev_normals) {
        cudaFree(dev_normals);
        dev_normals = NULL;
    }
    if (dev_albedo) {
        cudaFree(dev_albedo);
        dev_albedo = NULL;
    }
    if (dev_denoised) {
        cudaFree(dev_denoised);
        dev_denoised = NULL;
    }

    // Cleanup denoiser
    if (g_denoiser) {
        delete g_denoiser;
        g_denoiser = nullptr;
    }

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // === STOCHASTIC SAMPLED ANTIALIASING ===

        // Define the subdivision level (e.g., 2x2, 3x3, or 4x4 grid per pixel)
        // Higher values give better antialiasing but require more samples for convergence
        const int GRID_SIZE = 2; // Can be 2, 3, or 4 for 4, 9, or 16 samples per pixel
        const int CELLS_PER_PIXEL = GRID_SIZE * GRID_SIZE;

        // Create a random number generator for this pixel
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        // Determine which cell to sample for this iteration
        // This ensures we cycle through all cells over multiple iterations
        int cellIndex = iter % CELLS_PER_PIXEL;
        int cellX = cellIndex % GRID_SIZE;
        int cellY = cellIndex / GRID_SIZE;

        // Calculate the jittered offset within the pixel
        // Each cell has size (1.0 / GRID_SIZE) in pixel coordinates
        float cellSize = 1.0f / (float)GRID_SIZE;

        // Random offset within the selected cell
        float jitterX = (cellX + u01(rng)) * cellSize;
        float jitterY = (cellY + u01(rng)) * cellSize;

        // Convert pixel coordinates to continuous coordinates with jitter
        // Subtract 0.5 to center the jitter pattern around pixel center
        float pixelX = (float)x + jitterX - 0.5f;
        float pixelY = (float)y + jitterY - 0.5f;

        // Generate ray direction through the jittered pixel position
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (pixelX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (pixelY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    ShadeableIntersection* intersections,
    Material* materials)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        int hit_material_id = -1;
        glm::vec2 hit_uv = glm::vec2(0.0f);  // ADD THIS

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside;
        glm::vec2 tmp_uv;  // Already exists
        int tmp_material_id;

        // Parse through all geometries
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                tmp_material_id = geom.materialid;
                tmp_uv = glm::vec2(0.5f);  // Default UV for primitives
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                tmp_material_id = geom.materialid;
                tmp_uv = glm::vec2(0.5f);  // Default UV for primitives
            }
            else if (geom.type == GLTF_MESH)
            {
                t = meshIntersectionTest(geom, triangles, pathSegment.ray,
                    tmp_intersect, tmp_normal, tmp_outside,
                    tmp_uv, tmp_material_id);
            }

            // Update closest intersection
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
                hit_material_id = tmp_material_id;
                hit_uv = tmp_uv;  // STORE UV COORDINATES
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // Record intersection
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material_id;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = hit_uv;  // STORE UV IN INTERSECTION
        }
    }
}

__global__ void computeIntersectionsBVH(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    BVHNode** bvhNodes,        // Array of BVH node pointers
    int** bvhTriangleIndices,  // Array of triangle index pointers
    ShadeableIntersection* intersections,
    Material* materials,
    bool useBVH)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        int hit_material_id = -1;
        glm::vec2 hit_uv = glm::vec2(0.0f);

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside;
        glm::vec2 tmp_uv;
        int tmp_material_id;

        // Parse through all geometries
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                tmp_material_id = geom.materialid;
                tmp_uv = glm::vec2(0.5f);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                tmp_material_id = geom.materialid;
                tmp_uv = glm::vec2(0.5f);
            }
            else if (geom.type == GLTF_MESH)
            {
                if (useBVH && geom.bvhIndex >= 0 && bvhNodes != nullptr) {
                    // Use BVH traversal
                    t = meshIntersectionTestBVH(
                        geom, triangles,
                        bvhNodes[geom.bvhIndex],
                        bvhTriangleIndices[geom.bvhIndex],
                        pathSegment.ray,
                        tmp_intersect, tmp_normal, tmp_outside,
                        tmp_uv, tmp_material_id
                    );
                }
                else {
                    // Fallback to linear search
                    t = meshIntersectionTest(
                        geom, triangles, pathSegment.ray,
                        tmp_intersect, tmp_normal, tmp_outside,
                        tmp_uv, tmp_material_id
                    );
                }
            }

            // Update closest intersection
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
                hit_material_id = tmp_material_id;
                hit_uv = tmp_uv;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // Record intersection
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material_id;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = hit_uv;
        }
    }
}


__device__ glm::vec4 getPixelFromTextureWithAlpha(const GPUTexture& texture, int x, int y) {
    int idx = (y * texture.width + x) * texture.components;

    if (texture.components == 3) {
        return glm::vec4(
            texture.data[idx] / 255.0f,
            texture.data[idx + 1] / 255.0f,
            texture.data[idx + 2] / 255.0f,
            1.0f  // Opaque for RGB textures
        );
    }
    else if (texture.components == 4) {
        return glm::vec4(
            texture.data[idx] / 255.0f,
            texture.data[idx + 1] / 255.0f,
            texture.data[idx + 2] / 255.0f,
            texture.data[idx + 3] / 255.0f  // Include alpha channel
        );
    }
    else if (texture.components == 1) {
        float val = texture.data[idx] / 255.0f;
        return glm::vec4(val, val, val, 1.0f);
    }
    return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f); // Magenta for error detection
}


__device__ glm::vec3 getPixelFromTexture(const GPUTexture& texture, int x, int y) {
    int idx = (y * texture.width + x) * texture.components;

    if (texture.components == 3) {
        return glm::vec3(
            texture.data[idx] / 255.0f,
            texture.data[idx + 1] / 255.0f,
            texture.data[idx + 2] / 255.0f
        );
    }
    else if (texture.components == 4) {
        return glm::vec3(
            texture.data[idx] / 255.0f,
            texture.data[idx + 1] / 255.0f,
            texture.data[idx + 2] / 255.0f
            // Ignoring alpha for now
        );
    }
    else if (texture.components == 1) {
        float val = texture.data[idx] / 255.0f;
        return glm::vec3(val, val, val);
    }
    return glm::vec3(1.0f, 0.0f, 1.0f); // Magenta for error detection
}

__device__ glm::vec4 sampleTextureWithAlpha(const GPUTexture& texture, glm::vec2 uv) {
    // Wrap UV coordinates
    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);

    // Convert to pixel coordinates
    float fx = uv.x * (texture.width - 1);
    float fy = uv.y * (texture.height - 1);

    // Bilinear interpolation
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, texture.width - 1);
    int y1 = min(y0 + 1, texture.height - 1);

    float wx = fx - x0;
    float wy = fy - y0;

    // Sample four neighboring pixels with alpha
    glm::vec4 p00 = getPixelFromTextureWithAlpha(texture, x0, y0);
    glm::vec4 p10 = getPixelFromTextureWithAlpha(texture, x1, y0);
    glm::vec4 p01 = getPixelFromTextureWithAlpha(texture, x0, y1);
    glm::vec4 p11 = getPixelFromTextureWithAlpha(texture, x1, y1);

    // Bilinear interpolation
    glm::vec4 p0 = p00 * (1.0f - wx) + p10 * wx;
    glm::vec4 p1 = p01 * (1.0f - wx) + p11 * wx;
    glm::vec4 result = p0 * (1.0f - wy) + p1 * wy;

    return result;
}

__device__ glm::vec3 sampleTextureClean(const GPUTexture& texture, glm::vec2 uv) {
    glm::vec4 color = sampleTextureWithAlpha(texture, uv);
    return glm::vec3(color.x, color.y, color.z);
}



// ===== DEVICE FUNCTIONS FOR ENVIRONMENT MAP =====

__device__ glm::vec3 sampleEnvironmentMap(const glm::vec3& direction, const EnvironmentMap& envMap)
{
    if (!envMap.enabled || envMap.data == nullptr) {
        return glm::vec3(0.0f);
    }

    // Convert direction to spherical coordinates
    // theta: angle from +Y axis (0 to PI)
    // phi: angle around Y axis from +X (0 to 2*PI)
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, direction.y)));
    float phi = atan2f(direction.z, direction.x);

    // Convert to UV coordinates [0, 1]
    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;

    // Clamp UV to valid range
    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));

    // Convert to pixel coordinates
    float fx = u * (envMap.width - 1);
    float fy = v * (envMap.height - 1);

    // Bilinear interpolation for smoother sampling
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, envMap.width - 1);
    int y1 = min(y0 + 1, envMap.height - 1);

    float wx = fx - x0;
    float wy = fy - y0;

    // Sample four neighboring pixels
    glm::vec3 p00 = envMap.data[y0 * envMap.width + x0];
    glm::vec3 p10 = envMap.data[y0 * envMap.width + x1];
    glm::vec3 p01 = envMap.data[y1 * envMap.width + x0];
    glm::vec3 p11 = envMap.data[y1 * envMap.width + x1];

    // Bilinear interpolation
    glm::vec3 p0 = p00 * (1.0f - wx) + p10 * wx;
    glm::vec3 p1 = p01 * (1.0f - wx) + p11 * wx;
    glm::vec3 result = p0 * (1.0f - wy) + p1 * wy;

    return result;
}

// ===== HELPER FUNCTIONS FOR MIS =====

// Power heuristic for MIS (balance heuristic with beta=2)
__device__ float powerHeuristic(float pdfA, float pdfB) {
    // FIREFLY FIX: Ensure non-zero PDFs
    pdfA = max(pdfA, 1e-8f);
    pdfB = max(pdfB, 1e-8f);

    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    return pdfA2 / (pdfA2 + pdfB2);
}

// Sample a point on a sphere
__device__ glm::vec3 sampleSphere(const Geom& geom, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float u = u01(rng);
    float v = u01(rng);

    float theta = 2.0f * PI * u;
    float phi = acos(1.0f - 2.0f * v);

    float radius = geom.scale.x * 0.5f; // Assuming uniform scale for sphere

    glm::vec3 local(
        radius * sin(phi) * cos(theta),
        radius * sin(phi) * sin(theta),
        radius * cos(phi)
    );

    return glm::vec3(geom.transform * glm::vec4(local, 1.0f));
}

// Sample a point on a box
__device__ glm::vec3 sampleBox(const Geom& geom, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Choose which face to sample
    float faceChoice = u01(rng) * 6.0f;
    int face = (int)faceChoice;

    float u = u01(rng) - 0.5f;
    float v = u01(rng) - 0.5f;

    glm::vec3 local;
    switch (face) {
    case 0: local = glm::vec3(0.5f, u, v); break;   // +X
    case 1: local = glm::vec3(-0.5f, u, v); break;  // -X
    case 2: local = glm::vec3(u, 0.5f, v); break;   // +Y
    case 3: local = glm::vec3(u, -0.5f, v); break;  // -Y
    case 4: local = glm::vec3(u, v, 0.5f); break;   // +Z
    default: local = glm::vec3(u, v, -0.5f); break; // -Z
    }

    return glm::vec3(geom.transform * glm::vec4(local, 1.0f));
}

// Sample a point on any light source
__device__ glm::vec3 sampleLight(const Geom& geom, thrust::default_random_engine& rng) {
    if (geom.type == SPHERE) {
        return sampleSphere(geom, rng);
    }
    else if (geom.type == CUBE) {
        return sampleBox(geom, rng);
    }
    return geom.translation; // Fallback to center
}

// ==== = ENVIRONMENT MAP IMPORTANCE SAMPLING HELPERS ==== =

// Convert direction to environment map UV coordinates
__device__ glm::vec2 directionToUV(const glm::vec3 & direction) {
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, direction.y)));
    float phi = atan2f(direction.z, direction.x);

    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;

    return glm::vec2(u, v);
}

// Convert UV coordinates to direction
__device__ glm::vec3 uvToDirection(float u, float v) {
    float phi = u * 2.0f * PI - PI;
    float theta = v * PI;

    float sinTheta = sinf(theta);
    return glm::vec3(
        sinTheta * cosf(phi),
        cosf(theta),
        sinTheta * sinf(phi)
    );
}

// Get the solid angle of a pixel in the environment map
__device__ float getPixelSolidAngle(int x, int y, int width, int height) {
    float v = (y + 0.5f) / height;
    float theta = v * PI;
    float sinTheta = sinf(theta);

    // Solid angle is proportional to sin(theta) for equirectangular projection
    float pixelArea = (2.0f * PI / width) * (PI / height);
    return pixelArea * sinTheta;
}

// Sample environment map with uniform sampling
__device__ glm::vec3 sampleEnvironmentUniform(
    const glm::vec3& normal,
    const EnvironmentMap& envMap,
    thrust::default_random_engine& rng,
    glm::vec3& outDirection,
    float& outPdf
) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Sample uniform direction on hemisphere
    float u = u01(rng);
    float v = u01(rng);

    float phi = 2.0f * PI * u;
    float cosTheta = v;  // Uniform in cos(theta) for hemisphere
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    // Local space direction
    glm::vec3 localDir(
        sinTheta * cosf(phi),
        cosTheta,
        sinTheta * sinf(phi)
    );

    // Transform to world space aligned with normal
    glm::vec3 tangent, bitangent;
    if (fabs(normal.x) < 0.9f) {
        tangent = glm::normalize(glm::cross(glm::vec3(1, 0, 0), normal));
    }
    else {
        tangent = glm::normalize(glm::cross(glm::vec3(0, 1, 0), normal));
    }
    bitangent = glm::cross(normal, tangent);

    outDirection = localDir.x * tangent + localDir.z * bitangent + localDir.y * normal;
    outPdf = 1.0f / (2.0f * PI);  // Uniform hemisphere PDF

    // Sample the environment map
    return sampleEnvironmentMap(outDirection, envMap);
}

// Compute PDF for a given direction in the environment map
__device__ float environmentPdf(const glm::vec3& direction, const EnvironmentMap& envMap) {
    // For uniform sampling
    return 1.0f / (2.0f * PI);

    // For importance sampling (would need precomputed CDFs):
    // glm::vec2 uv = directionToUV(direction);
    // int x = (int)(uv.x * envMap.width);
    // int y = (int)(uv.y * envMap.height);
    // return luminancePdf[y * envMap.width + x];
}

// ===== SHADING HELPER FUNCTIONS =====
__host__ __device__ void shadeDiffuse(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    glm::vec3 materialColor,
    thrust::default_random_engine rng
    ) {
    

    // Generate new ray direction using cosine-weighted sampling
    glm::vec3 wiW = calculateRandomDirectionInHemisphere(
        intersection.surfaceNormal, rng);

    // For cosine-weighted sampling with Lambertian BRDF:
    // The math simplifies to just multiplying by the material color
    pathSegment.color *= materialColor;

    // Set up the new ray
    glm::vec3 intersectionPoint = pathSegment.ray.origin +
        pathSegment.ray.direction * intersection.t;
    pathSegment.ray.origin = intersectionPoint +
        intersection.surfaceNormal * 0.001f;
    pathSegment.ray.direction = wiW;
}

// ===== MIS DIFFUSE SHADING WITH DIRECT LIGHTING =====
__device__ glm::vec3 clamp(const glm::vec3& v, const glm::vec3& min, const glm::vec3& max) {
    return glm::vec3(
        fminf(fmaxf(v.x, min.x), max.x),
        fminf(fmaxf(v.y, min.y), max.y),
        fminf(fmaxf(v.z, min.z), max.z)
    );
}

__device__ float clamp(float v, float min, float max) {
    return fminf(fmaxf(v, min), max);
}

// ===== PBR HELPER FUNCTIONS =====

// GGX/Trowbridge-Reitz Normal Distribution Function
__device__ float GGX_D(const glm::vec3& n, const glm::vec3& h, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH = fmaxf(0.0f, glm::dot(n, h));
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (alpha2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;

    return alpha2 / fmaxf(0.0001f, denom);
}

// Schlick-GGX Geometry Function (single term)
__device__ float GGX_G1(const glm::vec3& n, const glm::vec3& v, float roughness) {
    float alpha = roughness * roughness;
    float k = alpha / 2.0f; // For IBL, use (alpha * alpha) / 2.0f

    float NdotV = fmaxf(0.0f, glm::dot(n, v));
    float denom = NdotV * (1.0f - k) + k;

    return NdotV / fmaxf(0.0001f, denom);
}

// Smith's Geometry Function (combines two G1 terms)
__device__ float GGX_G(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float roughness) {
    return GGX_G1(n, v, roughness) * GGX_G1(n, l, roughness);
}

// Fresnel term using Schlick's approximation
__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    return F0 + (glm::vec3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

// Sample GGX distribution for importance sampling
__device__ glm::vec3 sampleGGX(const glm::vec3& normal, float roughness, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float u = u01(rng);
    float v = u01(rng);

    float alpha = roughness * roughness;

    // Sample in tangent space
    float phi = 2.0f * PI * u;
    float cosTheta = sqrtf((1.0f - v) / (1.0f + (alpha * alpha - 1.0f) * v));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    glm::vec3 h_tangent(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );

    // Transform to world space
    glm::vec3 up = fabs(normal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 tangentX = glm::normalize(glm::cross(up, normal));
    glm::vec3 tangentY = glm::cross(normal, tangentX);

    return tangentX * h_tangent.x + tangentY * h_tangent.y + normal * h_tangent.z;
}

__device__ int binarySearch(const float* cdf, int size, float value) {
    int left = 0;
    int right = size - 1;

    while (left < right) {
        int mid = (left + right) / 2;
        if (cdf[mid] < value) {
            left = mid + 1;
        }
        else {
            right = mid;
        }
    }

    return left;
}

__device__ glm::vec3 sampleEnvironmentMapImportance(
    const EnvironmentMap& envMap,
    const EnvMapDistribution& distribution,
    thrust::default_random_engine& rng,
    glm::vec3& outDirection,
    float& outPdf)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float u = u01(rng);
    float v = u01(rng);

    // Sample row (theta) using marginal distribution
    int y = binarySearch(distribution.marginalCDF, envMap.height, v);
    y = min(y, envMap.height - 1);

    // Sample column (phi) using conditional distribution for this row
    const float* rowCDF = distribution.conditionalCDFs + y * envMap.width;
    int x = binarySearch(rowCDF, envMap.width, u);
    x = min(x, envMap.width - 1);

    // Convert pixel coordinates to direction
    float phi = (x + 0.5f) * 2.0f * PI / envMap.width - PI;
    float theta = (y + 0.5f) * PI / envMap.height;

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);

    outDirection = glm::normalize(glm::vec3(
        sinTheta * cosf(phi),
        cosTheta,
        sinTheta * sinf(phi)
    ));

    // Compute PDF
    // Get the luminance of the sampled pixel
    glm::vec3 color = envMap.data[y * envMap.width + x];
    float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;

    // PDF in image space
    float pdfImage = luminance * sinTheta;
    if (distribution.totalPower > 0) {
        pdfImage /= distribution.totalPower;
    }

    // Convert to solid angle measure
    float pixelArea = (2.0f * PI / envMap.width) * (PI / envMap.height);
    outPdf = pdfImage / (pixelArea * sinTheta);

    // Ensure valid PDF
    outPdf = fmaxf(outPdf, 1e-6f);

    return color;
}


__device__ glm::vec3 sampleEnvironmentMapImportance(
    const EnvironmentMap& envMap,
    thrust::default_random_engine& rng,
    glm::vec3& outDirection,
    float& outPdf)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (!envMap.distribution.marginalCDF || !envMap.distribution.conditionalCDFs) {
        // Fallback to uniform sampling if distribution not available
        outDirection = calculateRandomDirectionInHemisphere(glm::vec3(0, 1, 0), rng);
        outPdf = 1.0f / (2.0f * PI);
        return sampleEnvironmentMap(outDirection, envMap);
    }

    float u = u01(rng);
    float v = u01(rng);

    // Sample row (theta) using marginal distribution
    int y = binarySearch(envMap.distribution.marginalCDF, envMap.height, v);
    y = min(y, envMap.height - 1);

    // Sample column (phi) using conditional distribution for this row
    const float* rowCDF = envMap.distribution.conditionalCDFs + y * envMap.width;
    int x = binarySearch(rowCDF, envMap.width, u);
    x = min(x, envMap.width - 1);

    // Convert pixel coordinates to direction
    float phi = (x + 0.5f) * 2.0f * PI / envMap.width - PI;
    float theta = (y + 0.5f) * PI / envMap.height;

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);

    outDirection = glm::normalize(glm::vec3(
        sinTheta * cosf(phi),
        cosTheta,
        sinTheta * sinf(phi)
    ));

    // Get the color at this pixel
    glm::vec3 color = envMap.data[y * envMap.width + x];

    // Compute PDF
    float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;

    // PDF in image space
    float pdfImage = luminance * sinTheta;
    if (envMap.distribution.totalPower > 0) {
        pdfImage /= envMap.distribution.totalPower;
    }

    // Convert to solid angle measure
    float pixelArea = (2.0f * PI / envMap.width) * (PI / envMap.height);
    outPdf = pdfImage / (pixelArea * sinTheta);

    // Ensure valid PDF
    outPdf = fmaxf(outPdf, 1e-6f);

    return color;
}

__device__ float environmentPdfImportance(
    const glm::vec3& direction,
    const EnvironmentMap& envMap)
{
    if (!envMap.distribution.marginalCDF || !envMap.distribution.conditionalCDFs) {
        // Fallback to uniform PDF
        return 1.0f / (2.0f * PI);
    }

    // Convert direction to UV coordinates
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, direction.y)));
    float phi = atan2f(direction.z, direction.x);

    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;

    // Get pixel coordinates
    int x = min((int)(u * envMap.width), envMap.width - 1);
    int y = min((int)(v * envMap.height), envMap.height - 1);

    // Get luminance of this pixel
    glm::vec3 color = envMap.data[y * envMap.width + x];
    float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;

    // Compute PDF
    float sinTheta = sinf(theta);
    float pdfImage = luminance * sinTheta;

    if (envMap.distribution.totalPower > 0) {
        pdfImage /= envMap.distribution.totalPower;
    }

    // Convert to solid angle measure
    float pixelArea = (2.0f * PI / envMap.width) * (PI / envMap.height);
    float pdf = pdfImage / (pixelArea * sinTheta);

    return fmaxf(pdf, 1e-6f);
}



// ==========================================
// BSSRDF HELPER FUNCTIONS
// ==========================================
__device__ glm::vec3 evaluateDipoleProfile(
    float r,  // distance
    const glm::vec3& sigma_a,  // absorption
    const glm::vec3& sigma_s_prime  // reduced scattering
) {
    glm::vec3 sigma_t_prime = sigma_a + sigma_s_prime;
    glm::vec3 sigma_tr = glm::sqrt(3.0f * sigma_a * sigma_t_prime);

    glm::vec3 z_r = glm::vec3(1.0f) / sigma_t_prime;
    glm::vec3 z_v = z_r * (1.0f + 4.0f / 3.0f * glm::vec3(1.44f));  // A=1.44 for IOR~1.3

    glm::vec3 d_r = glm::sqrt(z_r * z_r + r * r);
    glm::vec3 d_v = glm::sqrt(z_v * z_v + r * r);

    // Normalized diffusion profile
    glm::vec3 C_phi = glm::vec3(0.25f) / PI;

    glm::vec3 result = C_phi * (
        z_r * (sigma_tr + glm::vec3(1.0f) / d_r) * glm::exp(-sigma_tr * d_r) / (d_r * d_r) +
        z_v * (sigma_tr + glm::vec3(1.0f) / d_v) * glm::exp(-sigma_tr * d_v) / (d_v * d_v)
        );

    result = glm::clamp(result, glm::vec3(0.0f), glm::vec3(1.0f));

    return result;
}

// Convert material parameters to absorption/scattering coefficients
__device__ void computeSSCoefficients(
    const Material& material,
    glm::vec3& sigma_a,
    glm::vec3& sigma_s_prime
) {
    // More physically plausible coefficient calculation
    glm::vec3 A = material.subsurfaceColor;

    glm::vec3 safeRadius = glm::max(material.subsurfaceRadiusRGB * material.subsurfaceScale,
        glm::vec3(0.001f));

    sigma_s_prime = glm::vec3(1.0f) / safeRadius;

    // Absorption coefficient (based on subsurface color)
    // Darker colors absorb more
    sigma_a = sigma_s_prime * (glm::vec3(1.0f) - A) * 0.01f;  // Reduced from 0.1f

    sigma_a = glm::clamp(sigma_a, glm::vec3(0.001f), glm::vec3(10.0f));
    sigma_s_prime = glm::clamp(sigma_s_prime, glm::vec3(0.1f), glm::vec3(100.0f));
}

// Sample a point inside the medium for SSS
__device__ glm::vec3 sampleSSExitPoint(
    const glm::vec3& entryPoint,
    const glm::vec3& normal,
    const Material& material,
    thrust::default_random_engine& rng
) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Average radius for sampling
    float avgRadius = (material.subsurfaceRadiusRGB.x +
        material.subsurfaceRadiusRGB.y +
        material.subsurfaceRadiusRGB.z) / 3.0f;
    avgRadius *= material.subsurfaceScale;

    avgRadius = glm::clamp(avgRadius, 0.001f, 1.0f);

    // Sample distance with exponential falloff
    float u = u01(rng);
    float distance = -logf(1.0f - u * 0.9f) * avgRadius;  // Prevent log(0)
    distance = glm::clamp(distance, 0.001f, avgRadius * 3.0f);  // Reduced from 10x

    // Sample direction uniformly in hemisphere below surface
    float theta = 2.0f * PI * u01(rng);
    float phi = acosf(1.0f - u01(rng));

    glm::vec3 localDir = glm::vec3(
        sinf(phi) * cosf(theta),
        sinf(phi) * sinf(theta),
        cosf(phi)
    );

    // Build tangent space
    glm::vec3 tangent = (fabs(normal.x) > fabs(normal.y)) ?
        glm::normalize(glm::vec3(-normal.z, 0, normal.x)) :
        glm::normalize(glm::vec3(0, -normal.z, normal.y));
    glm::vec3 bitangent = glm::cross(normal, tangent);

    // Transform to world space (pointing into the surface)
    glm::vec3 worldDir = tangent * localDir.x + bitangent * localDir.y - normal * localDir.z;

    return entryPoint + glm::normalize(worldDir) * distance;
}


__device__ bool sampleSubsurfaceScatteringPath(
    PathSegment& pathSegment,
    const glm::vec3& intersectionPoint,
    const glm::vec3& normal,
    const Material& material,
    const glm::vec3& materialColor,
    thrust::default_random_engine& rng
) {
    // Only apply to non-metallic materials with SSS enabled
    if (material.subsurfaceEnabled == 0 || material.metallic > 0.5f) {
        return false;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    // Probability of taking SSS path (reduced to prevent over-contribution)
    float sssProb = 0.3f * (1.0f - material.metallic);
    sssProb = glm::clamp(sssProb, 0.1f, 0.3f);

    if (u01(rng) > sssProb) {
        return false;  // Don't take SSS path
    }

    // === SAMPLE EXIT POINT ===
    glm::vec3 exitPoint = sampleSSExitPoint(intersectionPoint, normal, material, rng);
    float distance = glm::length(exitPoint - intersectionPoint);

    // === EVALUATE BSSRDF ===
    glm::vec3 sigma_a, sigma_s_prime;
    computeSSCoefficients(material, sigma_a, sigma_s_prime);

    // Use dipole model
    glm::vec3 bssrdf = evaluateDipoleProfile(distance, sigma_a, sigma_s_prime);

    // === APPLY TRANSMISSION THROUGH MEDIUM ===
    // Beer-Lambert law for absorption
    glm::vec3 transmittance = glm::exp(-sigma_a * distance);
    transmittance = glm::clamp(transmittance, glm::vec3(0.0f), glm::vec3(1.0f));

    // === CALCULATE THROUGHPUT ===
    // Combine all factors with proper normalization
    glm::vec3 throughput = bssrdf * transmittance * material.subsurfaceColor;

    // Apply material color but don't double-multiply
    throughput = throughput * glm::mix(glm::vec3(1.0f), materialColor, 0.5f);

    // === IMPORTANCE COMPENSATION ===
    // Compensate for sampling probability but prevent energy explosion
    throughput *= (1.0f / sssProb);

    // === ENERGY CONSERVATION ===
    float maxComponent = fmaxf(fmaxf(throughput.r, throughput.g), throughput.b);
    if (maxComponent > 2.0f) {
        throughput *= 2.0f / maxComponent;  // Normalize if too bright
    }

    // === APPLY TO PATH ===
    pathSegment.color *= throughput;

    // === SAMPLE NEW DIRECTION ===
    // Exit direction is diffuse-like (cosine-weighted hemisphere)
    glm::vec3 exitNormal = normal;  // Simplified - use entry normal

    // Sample cosine-weighted hemisphere
    float r1 = u01(rng);
    float r2 = u01(rng);
    float sinTheta = sqrtf(r1);
    float cosTheta = sqrtf(1.0f - r1);
    float phi = 2.0f * PI * r2;

    glm::vec3 localDir = glm::vec3(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );

    // Transform to world space
    glm::vec3 tangent = (fabs(exitNormal.x) > fabs(exitNormal.y)) ?
        glm::normalize(glm::vec3(-exitNormal.z, 0, exitNormal.x)) :
        glm::normalize(glm::vec3(0, -exitNormal.z, exitNormal.y));
    glm::vec3 bitangent = glm::cross(exitNormal, tangent);

    glm::vec3 worldDir = tangent * localDir.x + bitangent * localDir.y + exitNormal * localDir.z;

    // === UPDATE RAY ===
    pathSegment.ray.origin = exitPoint + exitNormal * 0.001f;
    pathSegment.ray.direction = glm::normalize(worldDir);
    pathSegment.remainingBounces--;

    return true;  // SSS path was taken
}



// ===== MAIN PBR SHADING FUNCTION =====

__device__ void shadePBR(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const Material& material,
    const glm::vec3& texturedColor,
    float textureAlpha, 
    Geom* geoms,
    int num_geoms,
    Material* materials,
    LightInfo* lights,
    int num_lights,
    EnvironmentMap& envMap,
    thrust::default_random_engine& rng
) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 intersectionPoint = pathSegment.ray.origin +
        pathSegment.ray.direction * intersection.t;
    glm::vec3 normal = intersection.surfaceNormal;

    if (material.subsurfaceEnabled > 0 && material.metallic < 0.5f) {
        if (sampleSubsurfaceScatteringPath(
            pathSegment, intersectionPoint, normal,
            material, texturedColor, rng)) {
            // SSS path was taken, exit early
            return;
        }
    }

    glm::vec3 wo = -pathSegment.ray.direction;

    // Material properties
    glm::vec3 albedo = texturedColor;
    float roughness = glm::clamp(material.roughness, 0.02f, 1.0f);
    float metallic = material.metallic;

    // Combine material transparency with texture alpha
    // Use multiplicative blending for proper transparency stacking
    float combinedTransparency = material.transparency;
    if (textureAlpha < 1.0f) {
        // Convert alpha to transparency and combine
        float textureTransparency = 1.0f - textureAlpha;
        combinedTransparency = 1.0f - ((1.0f - combinedTransparency) * (1.0f - textureTransparency));
    }

    // Calculate F0 (reflectance at normal incidence)
    glm::vec3 F0 = glm::vec3(0.04f);
    F0 = glm::mix(F0, albedo, metallic);

    // Handle transparency/transmission
    if (combinedTransparency > 0.0f && u01(rng) < combinedTransparency) {
        // Ray passes through the surface
        float ior = material.indexOfRefraction > 0 ? material.indexOfRefraction : 1.5f;

        // Check if we need refraction
        bool entering = glm::dot(normal, wo) > 0;
        glm::vec3 n = entering ? normal : -normal;
        float eta = entering ? (1.0f / ior) : ior;

        // Fresnel for transmission
        float cosTheta = glm::dot(n, wo);
        float k = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);

        glm::vec3 newDirection;
        if (k < 0.0f || roughness > 0.8f) {
            // Total internal reflection or very rough - just pass through
            newDirection = pathSegment.ray.direction;
            pathSegment.ray.origin = intersectionPoint - n * 0.001f;
        }
        else {
            // Refract
            newDirection = glm::normalize(
                eta * (-wo) + (eta * cosTheta - sqrtf(k)) * n
            );
            pathSegment.ray.origin = intersectionPoint - n * 0.001f;
        }

        // Add some roughness-based scattering for translucent materials
        if (roughness > 0.1f && roughness < 0.8f) {
            glm::vec3 scatter = glm::vec3(
                u01(rng) - 0.5f,
                u01(rng) - 0.5f,
                u01(rng) - 0.5f
            ) * roughness * 0.2f;
            newDirection = glm::normalize(newDirection + scatter);
        }

        pathSegment.ray.direction = newDirection;
        // Tint by material color with reduced influence for transparency
        pathSegment.color *= glm::mix(glm::vec3(1.0f), albedo, 1.0f - combinedTransparency);
        return;
    }

    // Opaque material - use PBR BRDF

    // Direct lighting component for MIS
    const float MIN_PDF = 1e-6f;
    const float MAX_CONTRIBUTION = 20.0f;
    glm::vec3 directLight(0.0f);

    // Sample lights for direct lighting
    if (num_lights > 0) {
        // Pick a random light
        int lightIdx = (int)(u01(rng) * num_lights);
        if (lightIdx >= num_lights) lightIdx = num_lights - 1;

        Geom& lightGeom = geoms[lights[lightIdx].geomIdx];
        glm::vec3 lightPos = sampleLight(lightGeom, rng);
        glm::vec3 wi = glm::normalize(lightPos - intersectionPoint);

        // Shadow ray check
        Ray shadowRay;
        shadowRay.origin = intersectionPoint + normal * 0.001f;
        shadowRay.direction = wi;

        float distToLight = glm::length(lightPos - intersectionPoint);
        bool visible = true;

        // Check for occlusion
        for (int i = 0; i < num_geoms; i++) {
            if (i == lights[lightIdx].geomIdx) continue;

            glm::vec3 temp_intersect, temp_normal;
            bool temp_outside;
            float t = -1;

            if (geoms[i].type == CUBE) {
                t = boxIntersectionTest(geoms[i], shadowRay, temp_intersect, temp_normal, temp_outside);
            }
            else if (geoms[i].type == SPHERE) {
                t = sphereIntersectionTest(geoms[i], shadowRay, temp_intersect, temp_normal, temp_outside);
            } 
   //         else if (geoms[i].type == GLTF_MESH) {
   //             glm::vec2 temp_uv;
   //             int temp_material_id;

   //             t = meshIntersectionTestBVH(
   //                 geoms[i], triangles,
   //                 bvhNodes[geoms[i].bvhIndex],
   //                 bvhTriangleIndices[geoms[i].bvhIndex],
   //                 shadowRay,
   //                 temp_intersect, temp_normal, temp_outside,
   //                 temp_uv, temp_material_id
   //             );
			//}

            if (t > 0.001f && t < distToLight - 0.001f) {
                visible = false;
                break;
            }
        }

        if (visible) {
            // Calculate PBR BRDF for direct light
            glm::vec3 h = glm::normalize(wi + wo);
            float NdotL = fmaxf(0.0f, glm::dot(normal, wi));
            float NdotV = fmaxf(0.0f, glm::dot(normal, wo));
            float NdotH = fmaxf(0.0f, glm::dot(normal, h));
            float VdotH = fmaxf(0.0f, glm::dot(wo, h));

            // Calculate BRDF components
            float D = GGX_D(normal, h, roughness);
            float G = GGX_G(normal, wo, wi, roughness);
            glm::vec3 F = fresnelSchlick(VdotH, F0);

            // Cook-Torrance BRDF
            glm::vec3 numerator = D * G * F;
            float denominator = 4.0f * NdotV * NdotL;
            glm::vec3 specular = numerator / fmaxf(0.001f, denominator);

            // Diffuse component (only for dielectrics)
            glm::vec3 kS = F; // Specular contribution
            glm::vec3 kD = glm::vec3(1.0f) - kS; // Diffuse contribution
            kD *= 1.0f - metallic; // Metals have no diffuse

            glm::vec3 diffuse = kD * albedo / PI;

            // Combine diffuse and specular
            glm::vec3 brdf = diffuse + specular;


            // Add light contribution
            Material& lightMat = materials[lightGeom.materialid];
            directLight += brdf * lightMat.color * lightMat.emittance * NdotL / (distToLight * distToLight);
        }
    }

    // Indirect lighting - importance sample the BRDF

    // Decide between diffuse and specular based on metallic and roughness
    float specularProbability = 0.5f + 0.5f * metallic; // More likely to sample specular for metals

    if (u01(rng) < specularProbability) {
        // Sample specular lobe using GGX importance sampling
        glm::vec3 h = sampleGGX(normal, roughness, rng);
        glm::vec3 wi = glm::reflect(-wo, h);

        // Make sure the sampled direction is in the hemisphere
        if (glm::dot(wi, normal) > 0.0f) {
            float NdotL = glm::dot(normal, wi);
            float NdotV = fmaxf(0.0f, glm::dot(normal, wo));
            float VdotH = fmaxf(0.0f, glm::dot(wo, h));

            // Calculate Fresnel
            glm::vec3 F = fresnelSchlick(VdotH, F0);

            // For metals, multiply by albedo; for dielectrics, use white
            glm::vec3 specColor = glm::mix(glm::vec3(1.0f), albedo, metallic);

            // Weight by fresnel and compensate for probability
            pathSegment.color *= specColor * F / specularProbability;

            // Set up new ray
            pathSegment.ray.origin = intersectionPoint + normal * 0.001f;
            pathSegment.ray.direction = wi;
            pathSegment.prevIsSpecular = true;
        }
        else {
            // Terminate if we sampled below the horizon
            pathSegment.remainingBounces = 0;
            pathSegment.color = glm::vec3(0.0f);
        }
    }
    else {
        // Sample diffuse lobe (only for non-metals)
        if (metallic < 1.0f) {
            glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);

            // Diffuse contribution
            glm::vec3 diffuseColor = albedo * (1.0f - metallic);

            // Compensate for probability
            pathSegment.color *= diffuseColor / (1.0f - specularProbability);

            // Set up new ray
            pathSegment.ray.origin = intersectionPoint + normal * 0.001f;
            pathSegment.ray.direction = wi;
            pathSegment.prevIsSpecular = false;
        }
        else {
            // Pure metal with no diffuse - terminate
            pathSegment.remainingBounces = 0;
            pathSegment.color = glm::vec3(0.0f);
        }
    }

    // Add direct lighting contribution
    directLight = clamp(directLight, glm::vec3(0.0f), glm::vec3(MAX_CONTRIBUTION));
    pathSegment.color += directLight;
}



__device__ void shadeDiffuseMIS(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    glm::vec3 materialColor,
    Geom* geoms,
    int num_geoms,
    Material* materials,
    LightInfo* lights,
    int num_lights,
    EnvironmentMap& envMap,  // Now includes distribution
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 intersectionPoint = pathSegment.ray.origin +
        pathSegment.ray.direction * intersection.t;
    glm::vec3 normal = intersection.surfaceNormal;

    const float MIN_PDF = 1e-6f;
    const float MAX_CONTRIBUTION = 20.0f;

    glm::vec3 totalLight(0.0f);

    // Choose sampling strategy: 0 = light, 1 = BRDF, 2 = environment
    float strategyChoice = u01(rng);
    int strategy;

    if (num_lights > 0 && envMap.enabled) {
        // All three strategies available
        if (strategyChoice < 0.33f) strategy = 0;      // Light sampling
        else if (strategyChoice < 0.66f) strategy = 1; // BRDF sampling
        else strategy = 2;                             // Environment sampling
    }
    else if (num_lights > 0) {
        // Only light and BRDF sampling
        strategy = (strategyChoice < 0.5f) ? 0 : 1;
    }
    else if (envMap.enabled) {
        // Only BRDF and environment sampling
        strategy = (strategyChoice < 0.5f) ? 1 : 2;
    }
    else {
        // Only BRDF sampling
        strategy = 1;
    }

    if (strategy == 0 && num_lights > 0) {
        // === LIGHT SAMPLING ===
        int lightIdx = min((int)(u01(rng) * num_lights), num_lights - 1);
        LightInfo& lightInfo = lights[lightIdx];
        Geom& lightGeom = geoms[lightInfo.geomIdx];
        Material& lightMat = materials[lightGeom.materialid];

        glm::vec3 lightPoint = sampleLight(lightGeom, rng);
        glm::vec3 wi = lightPoint - intersectionPoint;
        float dist = glm::length(wi);

        if (dist > 0.01f) {
            wi /= dist;
            float NdotL = glm::dot(normal, wi);

            if (NdotL > 0.0f) {
                // === SHADOW TEST (SIMPLIFIED) ===
                Ray shadowRay;
                shadowRay.origin = intersectionPoint + normal * 0.001f;
                shadowRay.direction = wi;

                bool visible = true;

                // Check all geometry for occlusion
                for (int geomIdx = 0; geomIdx < num_geoms; geomIdx++) {
                    // Skip the light source itself
                    if (geomIdx == lightInfo.geomIdx) continue;

                    Geom& occluder = geoms[geomIdx];

                    glm::vec3 temp_intersect;
                    glm::vec3 temp_normal;
                    bool temp_outside;
                    float t = -1.0f;

                    // Test intersection based on geometry type
                    if (occluder.type == SPHERE) {
                        t = sphereIntersectionTest(
                            occluder,
                            shadowRay,
                            temp_intersect,
                            temp_normal,
                            temp_outside
                        );
                    }
                    else if (occluder.type == CUBE) {
                        t = boxIntersectionTest(
                            occluder,
                            shadowRay,
                            temp_intersect,
                            temp_normal,
                            temp_outside
                        );
                    }

                    // Check if this object blocks the light
                    if (t > 0.001f && t < dist - 0.001f) {
                        visible = false;
                        break;
                    }
                }

                if (visible) {
                    // Compute contribution with MIS
                    glm::vec3 lightNormal = glm::normalize(lightPoint - lightGeom.translation);
                    float NdotL_light = fmaxf(0.0f, glm::dot(-wi, lightNormal));

                    // PDFs
                    float pdfLight = 1.0f / (fmaxf(lightInfo.area, 0.01f) * num_lights);
                    float pdfBRDF = NdotL / PI;
                    float pdfEnv = envMap.enabled ?
                        environmentPdfImportance(wi, envMap) : 0.0f;

                    // MIS weight using power heuristic
                    float sumPdf = pdfLight + pdfBRDF + pdfEnv;
                    float weight = pdfLight / fmaxf(sumPdf, MIN_PDF);

                    glm::vec3 Le = lightMat.color * lightMat.emittance;
                    glm::vec3 f = materialColor / PI;
                    float G = NdotL * NdotL_light / (dist * dist);

                    // Compensate for strategy selection probability
                    float strategyProb = (num_lights > 0 && envMap.enabled) ? 0.33f :
                        (num_lights > 0 || envMap.enabled) ? 0.5f : 1.0f;

                    totalLight += weight * Le * f * G * (float)num_lights / (strategyProb * fmaxf(pdfLight, MIN_PDF));
                }
            }
        }
    }
    else if (strategy == 2 && envMap.enabled) {
        // === ENVIRONMENT SAMPLING ===
        glm::vec3 envDir;
        float envPdf;
        glm::vec3 envColor = sampleEnvironmentMapImportance(envMap, rng, envDir, envPdf);

        float NdotL = glm::dot(normal, envDir);

        if (NdotL > 0.0f) {
            // Shadow test for environment
            Ray shadowRay;
            shadowRay.origin = intersectionPoint + normal * 0.001f;
            shadowRay.direction = envDir;

            bool visible = true;
            for (int i = 0; i < num_geoms && visible; i++) {
                glm::vec3 tmp_i, tmp_n;
                bool tmp_o;
                float t = -1.0f;

                if (geoms[i].type == CUBE)
                    t = boxIntersectionTest(geoms[i], shadowRay, tmp_i, tmp_n, tmp_o);
                else if (geoms[i].type == SPHERE)
                    t = sphereIntersectionTest(geoms[i], shadowRay, tmp_i, tmp_n, tmp_o);

                visible = (t < 0.001f);
            }

            if (visible) {
                // Compute contribution with MIS
                float pdfBRDF = NdotL / PI;
                float pdfLight = 0.0f; // No discrete lights for this direction

                // MIS weight
                float sumPdf = envPdf + pdfBRDF;
                float weight = envPdf / fmaxf(sumPdf, MIN_PDF);

                glm::vec3 f = materialColor / PI;

                // Compensate for strategy selection probability
                float strategyProb = (num_lights > 0 && envMap.enabled) ? 0.33f : 0.5f;

                totalLight += weight * envColor * f * NdotL / (strategyProb * fmaxf(envPdf, MIN_PDF));
            }
        }
    }

    // === BRDF SAMPLING (always do this for indirect) ===
    glm::vec3 brdfDir = calculateRandomDirectionInHemisphere(normal, rng);

    // Clamp total contribution
    totalLight = glm::clamp(totalLight, glm::vec3(0.0f), glm::vec3(MAX_CONTRIBUTION));

    // Apply direct lighting and prepare for next bounce
    pathSegment.color *= materialColor + totalLight;

    // Set up next ray for indirect lighting
    pathSegment.ray.origin = intersectionPoint + normal * 0.001f;
    pathSegment.ray.direction = brdfDir;
}

__host__ __device__ void shadeSpecular(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    glm::vec3 materialColor
    ) 
{
    // Perfectly specular reflection direction
    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec3 incident = pathSegment.ray.direction;
    glm::vec3 reflected = glm::normalize(incident - 2.0f * glm::dot(incident, normal) * normal);
    // For perfect specular reflection with ideal mirror BRDF:
    // The math simplifies to just multiplying by the material color
    pathSegment.color *= materialColor;
    // Set up the new ray
    glm::vec3 intersectionPoint = pathSegment.ray.origin +
        pathSegment.ray.direction * intersection.t;
    pathSegment.ray.origin = intersectionPoint +
        intersection.surfaceNormal * 0.001f;
    pathSegment.ray.direction = reflected;
}

__host__ __device__ float shlickFresnel(float cosTheta, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosTheta), 5.0f);
}

__host__ __device__ void shadeRefractive(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    glm::vec3 materialColor,
    float ior,
	thrust::default_random_engine& rng
)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 normal = intersection.surfaceNormal;
	glm::vec3 incident = glm::normalize(pathSegment.ray.direction);

	float cosTheta = glm::dot(incident, normal);
	bool entering = cosTheta < 0.0f;

    float etaI, etaT;
	if (entering) {
        etaI = 1.0f; // air
        etaT = ior;
        cosTheta = -cosTheta; 
    }
    else {
        etaI = ior;
        etaT = 1.0f; // air
        normal = -normal;
    }

	float eta = etaI / etaT;

	float sin2ThetaT = eta * eta * (1.0f - cosTheta * cosTheta);

	glm::vec3 newDirection;

    if (sin2ThetaT > 1.0f) {
        // Total internal reflection
		newDirection = glm::normalize(incident - 2.0f * glm::dot(incident, normal) * normal);
    }
    else {
        float cosThetaT = sqrtf(1.0f - sin2ThetaT);
		float fresnelReflectance = shlickFresnel(entering ? cosTheta : cosThetaT, eta);

        if (u01(rng) < fresnelReflectance) {
			// Reflect
			newDirection = glm::reflect(incident, normal);
        }
        else {
			newDirection = glm::normalize(eta * incident + (eta * cosTheta - cosThetaT) * normal);
        }
	}

	pathSegment.color *= materialColor;

	// Set up the new ray
    glm::vec3 intersectionPoint = pathSegment.ray.origin +
		pathSegment.ray.direction * intersection.t;

    pathSegment.ray.origin = intersectionPoint + newDirection * 0.001f;
	pathSegment.ray.direction = newDirection;
}

__global__ void extractMaterialIds(
    int num_paths,
    ShadeableIntersection* intersections,
    int* material_ids,
    int* path_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    // Use material ID for sorting (use large number for no intersection)
    material_ids[idx] = (intersections[idx].t > 0.0f) ?
        intersections[idx].materialId :
        INT_MAX;
    path_indices[idx] = idx;
}

__global__ void reorderByMaterial(
    int num_paths,
    PathSegment* paths_in,
    PathSegment* paths_out,
    ShadeableIntersection* intersections_in,
    ShadeableIntersection* intersections_out,
    int* sorted_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    int src_idx = sorted_indices[idx];
    paths_out[idx] = paths_in[src_idx];
    intersections_out[idx] = intersections_in[src_idx];
}

__device__ float computeLuminance(const glm::vec3& color) {
    // Standard luminance formula (Rec. 709)
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

// ===== MAIN SHADING KERNEL =====
__global__ void shadeMaterialMIS(
    int iter,
    int currentBounce,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    GPUTexture* textures,    // ADD THIS
    int num_textures,        // ADD THIS
    Geom* geoms,
    int num_geoms,
    LightInfo* lights,
    int num_lights,
    EnvironmentMap envMap,
    bool firstIter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    if (pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    thrust::default_random_engine rng = makeSeededRandomEngine(
        iter, idx, pathSegments[idx].remainingBounces);

    if (currentBounce >= RR_START_BOUNCE) {
        PathSegment& path = pathSegments[idx];
        float throughput = computeLuminance(path.color);
        float survivalProb = glm::clamp(throughput, RR_SURVIVAL_MIN, RR_SURVIVAL_MAX);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) >= survivalProb) {
            // Terminate path
            path.remainingBounces = 0;
            path.color = glm::vec3(0.0f);
            return;
        }
        else {
            // Path survives - compensate
            path.color /= survivalProb;
        }
    }

    if (intersection.t > 0.0f) {
        Material material = materials[intersection.materialId];

        glm::vec3 materialColor = material.color;
        float textureAlpha = 1.0f;  // Default to opaque


        // Sample base color texture if available
        if (material.baseColorTextureIdx >= 0 && material.baseColorTextureIdx < num_textures) {
            glm::vec4 sampledColor = sampleTextureWithAlpha(
                textures[material.baseColorTextureIdx],
                intersection.uv
            );
            materialColor = glm::vec3(sampledColor.x, sampledColor.y, sampledColor.z);
            textureAlpha = sampledColor.w;  // Extract alpha channel

            // Apply material color as a tint (multiply)
            materialColor *= material.color;
        }

        // Sample metallic-roughness texture if available
        if (material.metallicRoughnessTextureIdx >= 0 && material.metallicRoughnessTextureIdx < num_textures) {
            glm::vec3 metallicRoughness = sampleTextureClean(
                textures[material.metallicRoughnessTextureIdx], intersection.uv);
            // In GLTF: Blue = metallic, Green = roughness
            material.metallic *= metallicRoughness.z;
            material.roughness *= metallicRoughness.y;
        }

        // Sample emissive texture if available
        if (material.emissiveTextureIdx >= 0 && material.emissiveTextureIdx < num_textures) {
            glm::vec3 emissive = sampleTextureClean(textures[material.emissiveTextureIdx], intersection.uv);
            // If there's emissive texture, treat as light source
            if (glm::length(emissive) > 0.0f) {
                pathSegments[idx].color *= emissive * material.emissiveFactor;
                pathSegments[idx].remainingBounces = 0;
                return;
            }
        }

        // Handle light sources
        if (material.emittance > 0.0f) {
            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        pathSegments[idx].remainingBounces--;

        if (pathSegments[idx].remainingBounces <= 0) {
            pathSegments[idx].color = glm::vec3(0.0f);
            return;
        }

        

        MaterialType mType = material.type;
        switch (mType) {
        case DIFFUSE:
            pathSegments[idx].prevIsSpecular = false;
            shadeDiffuseMIS(pathSegments[idx], intersection, materialColor,
                geoms, num_geoms, materials, lights, num_lights, envMap, rng);
            break;

        case SPECULAR:
            pathSegments[idx].prevIsSpecular = true;
            shadeSpecular(pathSegments[idx], intersection, materialColor);
            break;

        case REFRACTIVE:
            pathSegments[idx].prevIsSpecular = true;
            shadeRefractive(pathSegments[idx], intersection, materialColor,
                material.indexOfRefraction, rng);
            break;

        case PBR:
            // PBR materials can be both specular and diffuse depending on parameters
            // Consider it specular if it's smooth and metallic
            pathSegments[idx].prevIsSpecular = (material.roughness < 0.1f && material.metallic > 0.5f);

            shadePBR(pathSegments[idx], intersection, material, materialColor,
                textureAlpha,  // Pass texture alpha
                geoms, num_geoms, materials, lights, num_lights, envMap, rng);
            break;

        default:
            pathSegments[idx].prevIsSpecular = false;
            break;
        }
    }
    else {
        // Handle environment map or background
        if (envMap.enabled) {
            glm::vec3 envColor = sampleEnvironmentMap(pathSegments[idx].ray.direction, envMap);
            if (firstIter) {
                pathSegments[idx].color *= envColor;
            }
            else if (pathSegments[idx].prevIsSpecular) {
                pathSegments[idx].color *= envColor;
            }
            else {
                pathSegments[idx].color *= envColor * 0.5f;
            }
        }
        else {
            pathSegments[idx].color *= glm::vec3(0.0f);
        }
        pathSegments[idx].remainingBounces = 0;
    }
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* paths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment path = paths[index];
        // Only add to image if this path is terminated
        if (path.remainingBounces == 0) {
            image[path.pixelIndex] += path.color;
        }
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    if (USE_DENOISER && iter == 1) {
        cudaMemset(dev_normals, 0, pixelcount * sizeof(glm::vec3));
        if (DENOISE_WITH_ALBEDO) {
            cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
        }
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool firstIter = true;
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (USE_BVH && dev_bvh_nodes != nullptr) {
            computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_triangles,
                dev_bvh_nodes,
                dev_bvh_triangle_indices,
                dev_intersections,
                dev_materials,
                USE_BVH
                );
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_triangles,
                dev_intersections,
                dev_materials
                );
        }
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if MATERIAL_SORTING
        // === MATERIAL SORTING IMPLEMENTATION ===
        if (num_paths > 0) {
            // Extract material IDs for sorting
            dim3 extractBlocks = (num_paths + blockSize1d - 1) / blockSize1d;
            extractMaterialIds << <extractBlocks, blockSize1d >> > (
                num_paths, dev_intersections,
                dev_material_ids, dev_path_indices
                );
            checkCUDAError("extract material IDs");

            // Sort indices by material ID using thrust
            thrust::device_ptr<int> thrust_keys(dev_material_ids);
            thrust::device_ptr<int> thrust_values(dev_path_indices);
            thrust::sort_by_key(thrust_keys, thrust_keys + num_paths, thrust_values);
            checkCUDAError("sort by material");

            // Reorder paths and intersections based on sorted order
            reorderByMaterial << <extractBlocks, blockSize1d >> > (
                num_paths, dev_paths, dev_paths_sorted,
                dev_intersections, dev_intersections_sorted,
                dev_path_indices
                );
            checkCUDAError("reorder by material");

            // Swap pointers to use sorted data
            PathSegment* temp_paths = dev_paths;
            dev_paths = dev_paths_sorted;
            dev_paths_sorted = temp_paths;

            ShadeableIntersection* temp_intersections = dev_intersections;
            dev_intersections = dev_intersections_sorted;
            dev_intersections_sorted = temp_intersections;
        }
#endif

		// Capture auxiliary G-buffers for denoising
        if (USE_DENOISER && depth == 0) {  // Only on first bounce
            dim3 captureBlocks = (num_paths + blockSize1d - 1) / blockSize1d;
            captureNormalsAndAlbedo << <captureBlocks, blockSize1d >> > (
                num_paths,
                dev_paths,
                dev_intersections,
                dev_normals,
                DENOISE_WITH_ALBEDO ? dev_albedo : nullptr,
                dev_materials,
                iter,
                cam.resolution.x,
                cam.resolution.y
                );
            checkCUDAError("capture normals and albedo");
        }

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,     
            num_textures,     
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lights,
            num_lights,
            dev_environmentMap,
            firstIter
            );


		cudaDeviceSynchronize();

        if (firstIter) {
            firstIter = false;
		}

        dim3 numBlocksGather = (num_paths + blockSize1d - 1) / blockSize1d;
        gatherTerminatedPaths << <numBlocksGather, blockSize1d >> > (
            num_paths,
            dev_image,
            dev_paths
            );
        checkCUDAError("gather terminated paths");

        // --- Stream Compaction Stage ---
        PathSegment* new_end = thrust::remove_if(thrust::device,
            dev_paths,
            dev_paths + num_paths,
            is_terminated());

        int paths_before = num_paths;
        num_paths = new_end - dev_paths;

        // Check termination conditions
        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    bool shouldDenoise = USE_DENOISER &&
        g_denoiser &&
        g_denoiser->isInitialized() &&
        iter >= DENOISE_START_ITER &&
        (iter % DENOISE_FREQUENCY == 0);

    if (shouldDenoise) {
        // Copy current accumulated image to denoised buffer
        cudaMemcpy(dev_denoised, dev_image, pixelcount * sizeof(glm::vec3),
            cudaMemcpyDeviceToDevice);

        // Apply denoiser
        g_denoiser->denoise(
            dev_denoised,                              // Beauty buffer (input/output)
            DENOISE_WITH_NORMALS ? dev_normals : nullptr,  // Normal buffer 
            DENOISE_WITH_ALBEDO ? dev_albedo : nullptr,    // Albedo buffer 
            dev_denoised                               // Output buffer
        );

        // Send denoised result to PBO for display
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised);

        // Copy denoised result to host
        cudaMemcpy(hst_scene->state.image.data(), dev_denoised,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    else {
        // Send regular accumulated image to PBO
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

        // Retrieve image from GPU
        cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    //sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    //cudaMemcpy(hst_scene->state.image.data(), dev_image,
    //    pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
