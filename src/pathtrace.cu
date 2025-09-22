#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

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


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
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

    printf("Initialized %d light sources for MIS\n", num_lights);
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
    // Initialize environment map
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

        //printf("Environment map uploaded to GPU: %dx%d pixels, %.2f MB\n",
        //    scene->environmentMap.width, scene->environmentMap.height,
        //    envMapSize / (1024.0f * 1024.0f));
    }
    else
    {
        dev_environmentMap.data = nullptr;
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

    if (dev_lights != NULL) {
        cudaFree(dev_lights);
        dev_lights = NULL;
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

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
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
    ShadeableIntersection* intersections)
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

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
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

__device__ void shadeDiffuseMIS(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    glm::vec3 materialColor,
    Geom* geoms,
    int num_geoms,
    Material* materials,
    LightInfo* lights,
    int num_lights,
    thrust::default_random_engine& rng
) {
    if (num_lights == 0) {
        shadeDiffuse(pathSegment, intersection, materialColor, rng);
        return;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 intersectionPoint = pathSegment.ray.origin +
        pathSegment.ray.direction * intersection.t;
    glm::vec3 normal = intersection.surfaceNormal;

    glm::vec3 totalContribution(0.0f);

    // === DIRECT LIGHTING (Light Sampling) ===

    // Randomly select a light
    int lightIdx = (int)(u01(rng) * num_lights);
    lightIdx = min(lightIdx, num_lights - 1);
    LightInfo& lightInfo = lights[lightIdx];
    Geom& lightGeom = geoms[lightInfo.geomIdx];
    Material& lightMat = materials[lightGeom.materialid];

    // Sample a point on the light
    glm::vec3 lightPoint = sampleLight(lightGeom, rng);
    glm::vec3 toLight = lightPoint - intersectionPoint;
    float distToLight = glm::length(toLight);

    // FIREFLY FIX #1: Skip if too close to avoid numerical issues
    if (distToLight < 0.01f) {
        shadeDiffuse(pathSegment, intersection, materialColor, rng);
        return;
    }

    toLight = glm::normalize(toLight);

    // Check visibility (shadow ray)
    Ray shadowRay;
    shadowRay.origin = intersectionPoint + normal * 0.001f;
    shadowRay.direction = toLight;

    bool visible = true;
    glm::vec3 tmp_intersect, tmp_normal;
    bool tmp_outside;

    // Check intersection with all geometries
    for (int i = 0; i < num_geoms; i++) {
        if (i == lightInfo.geomIdx) continue; // Skip the light itself

        float t = -1.0f;
        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], shadowRay, tmp_intersect, tmp_normal, tmp_outside);
        }
        else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], shadowRay, tmp_intersect, tmp_normal, tmp_outside);
        }

        if (t > 0.0f && t < distToLight - 0.001f) {
            visible = false;
            break;
        }
    }

    if (visible) {
        // Compute light normal at sampled point (approximate)
        glm::vec3 lightNormal = glm::normalize(lightPoint - lightGeom.translation);

        // Geometric term
        float cosThetaLight = abs(glm::dot(-toLight, lightNormal));
        float cosThetaSurface = max(0.0f, glm::dot(toLight, normal));

        // FIREFLY FIX #2: Clamp distance squared to avoid extreme values
        float distSquaredClamped = max(distToLight * distToLight, 0.1f);
        float geometricTerm = cosThetaSurface * cosThetaLight / distSquaredClamped;

        // FIREFLY FIX #3: Ensure minimum area to avoid huge PDFs
        float safeArea = max(lightInfo.area, 0.01f);

        // PDFs for MIS
        float pdfLight = (1.0f / safeArea) * (1.0f / num_lights);
        float pdfBRDF = cosThetaSurface / PI;

        // FIREFLY FIX #4: Clamp PDFs to reasonable range
        pdfLight = clamp(pdfLight, 0.001f, 1000.0f);
        pdfBRDF = max(pdfBRDF, 0.001f);

        // FIREFLY FIX #5: Use balance heuristic for small lights (more stable)
        float misWeight;
        if (lightInfo.area < 0.1f) {
            // Balance heuristic for small lights
            misWeight = pdfLight / (pdfLight + pdfBRDF);
        }
        else {
            // Power heuristic for normal lights
            float pdfLight2 = pdfLight * pdfLight;
            float pdfBRDF2 = pdfBRDF * pdfBRDF;
            misWeight = pdfLight2 / (pdfLight2 + pdfBRDF2);
        }

        // Direct lighting contribution
        glm::vec3 lightEmission = lightMat.color * lightMat.emittance;
        glm::vec3 brdf = materialColor / PI;

        glm::vec3 contribution = misWeight * lightEmission * brdf * geometricTerm / pdfLight;

        // FIREFLY FIX #6: Clamp final contribution to prevent fireflies
        const float MAX_CONTRIBUTION = 10.0f;  // Tune this based on scenes
        contribution = clamp(contribution, glm::vec3(0.0f), glm::vec3(MAX_CONTRIBUTION));

        totalContribution += contribution;
    }

    // === INDIRECT LIGHTING (BRDF Sampling) ===

    // Sample the BRDF (cosine-weighted hemisphere)
    glm::vec3 wiW = calculateRandomDirectionInHemisphere(normal, rng);

    // Update path color with direct lighting contribution
    pathSegment.color *= materialColor + totalContribution;

    // Set up the new ray for indirect lighting
    pathSegment.ray.origin = intersectionPoint + normal * 0.001f;
    pathSegment.ray.direction = wiW;
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

// ===== MAIN SHADING KERNEL =====
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
	EnvironmentMap envMap,
    bool firstIter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    // Skip already terminated paths
    if (pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) // Ray hit something
    {
        

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        // Handle light sources
        if (material.emittance > 0.0f) {
            // Ray hit a light - accumulate emission and terminate
            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        // Handle diffuse materials
        // Decrement bounces first
        pathSegments[idx].remainingBounces--;

        // Check if we should continue
        if (pathSegments[idx].remainingBounces <= 0) {
            // Maximum depth reached without hitting light
            // The path contributes nothing (black)
            pathSegments[idx].color = glm::vec3(0.0f);
            return;
        }

        // Set up RNG with proper seed
        thrust::default_random_engine rng = makeSeededRandomEngine(
            iter, idx, pathSegments[idx].remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);

		MaterialType mType = material.type;

        switch (mType) {
        case DIFFUSE:
			pathSegments[idx].prevIsSpecular = false;
            // TODO: implement MIS
            shadeDiffuse(pathSegments[idx], intersection, materialColor, rng);
            break;

		case SPECULAR:
            pathSegments[idx].prevIsSpecular = true;
			shadeSpecular(pathSegments[idx], intersection, materialColor);
			break;

		case REFRACTIVE:
            pathSegments[idx].prevIsSpecular = true;
			shadeRefractive(pathSegments[idx], intersection, materialColor, material.indexOfRefraction, rng);
			break;

        default:
            pathSegments[idx].prevIsSpecular = false;
            shadeDiffuse(pathSegments[idx], intersection, materialColor, rng);
            break;
        }

    }
    else {
        

        if (envMap.enabled) {
            glm::vec3 envColor = sampleEnvironmentMap(pathSegments[idx].ray.direction, envMap);
            if (firstIter) {
                // Direct visibility of environment
                pathSegments[idx].color *= envColor;
            }
            else if (pathSegments[idx].prevIsSpecular) {
                // Environment visible through reflection/refraction
                pathSegments[idx].color *= envColor;
            }
            else {
                // Diffuse bounce missed - could use ambient or black
                // Using environment as ambient light
                pathSegments[idx].color *= envColor * 0.5f; // Reduced contribution
            }
        }
        else {
            // No environment map - use black
            pathSegments[idx].color *= glm::vec3(0.0f);
        }

        //pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = 0;
    }
}

// ===== MODIFIED SHADING KERNEL =====

__global__ void shadeMaterialMIS(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
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

    if (intersection.t > 0.0f) {
        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

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

        thrust::default_random_engine rng = makeSeededRandomEngine(
            iter, idx, pathSegments[idx].remainingBounces);

        MaterialType mType = material.type;

        switch (mType) {
        case DIFFUSE:
            pathSegments[idx].prevIsSpecular = false;
            // Use MIS for diffuse materials
            shadeDiffuseMIS(pathSegments[idx], intersection, materialColor,
                geoms, num_geoms, materials, lights, num_lights, rng);
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

        default:
            pathSegments[idx].prevIsSpecular = false;
            shadeDiffuseMIS(pathSegments[idx], intersection, materialColor,
                geoms, num_geoms, materials, lights, num_lights, rng);
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
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

   //     shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
   //         iter,
   //         num_paths,
   //         dev_intersections,
   //         dev_paths,
   //         dev_materials,
			//dev_environmentMap,
   //         firstIter
   //     );
        shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
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

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
