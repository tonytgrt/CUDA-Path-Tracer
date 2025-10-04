#include "optixDenoiser.h"
#include <optix_function_table_definition.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernels 
extern "C" void launchConvertFloat3ToFloat4(
    glm::vec3* src, float4* dst, unsigned int numPixels);

extern "C" void launchConvertFloat4ToFloat3(
    float4* src, glm::vec3* dst, unsigned int numPixels);

void OptiXDenoiser::context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    if (level < 4) {
        std::cerr << "[OptiX][" << tag << "]: " << message << std::endl;
    }
}

OptiXDenoiser::OptiXDenoiser()
{
}

OptiXDenoiser::~OptiXDenoiser()
{
    cleanup();
}

void OptiXDenoiser::init(unsigned int width, unsigned int height, bool useNormals, bool useAlbedo)
{
    if (m_initialized) {
        cleanup();
    }

    m_imageWidth = width;
    m_imageHeight = height;
    m_useNormals = useNormals;
    m_useAlbedo = useAlbedo;

    // Initialize CUDA (should already be initialized by path tracer)
    CUDA_CHECK(cudaFree(nullptr));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX context
    CUcontext cuCtx = 0;  // Use current CUDA context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;  // Only log errors
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));

    // Create denoiser
    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = m_useAlbedo ? 1 : 0;
    denoiserOptions.guideNormal = m_useNormals ? 1 : 0;

    OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;
    OPTIX_CHECK(optixDenoiserCreate(m_context, modelKind, &denoiserOptions, &m_denoiser));

    // Compute memory requirements
    OptixDenoiserSizes denoiserSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageWidth, m_imageHeight, &denoiserSizes));

    m_scratchSize = denoiserSizes.withoutOverlapScratchSizeInBytes;
    m_stateSize = denoiserSizes.stateSizeInBytes;

    // Allocate denoiser state and scratch memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_scratch), m_scratchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state), m_stateSize));

    // Allocate memory for HDR intensity and average color
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_intensity), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_avgColor), 3 * sizeof(float)));

    // Setup denoiser
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        0,  // Use default stream
        m_imageWidth,
        m_imageHeight,
        m_state,
        m_stateSize,
        m_scratch,
        m_scratchSize
    ));

    // Allocate device buffers for image data (float4 format)
    size_t pixelCount = m_imageWidth * m_imageHeight;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_beautyBuffer), pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_outputBuffer), pixelCount * sizeof(float4)));

    if (m_useNormals) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_normalBuffer), pixelCount * sizeof(float4)));
    }
    if (m_useAlbedo) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_albedoBuffer), pixelCount * sizeof(float4)));
    }

    m_initialized = true;
    std::cout << "OptiX Denoiser initialized for " << width << "x" << height
        << " images (Normals: " << (useNormals ? "ON" : "OFF")
        << ", Albedo: " << (useAlbedo ? "ON" : "OFF") << ")" << std::endl;
}

void OptiXDenoiser::convertFloat3ToFloat4(void* src, CUdeviceptr dst, unsigned int numPixels)
{
    // Call the external CUDA kernel launcher
    launchConvertFloat3ToFloat4(
        reinterpret_cast<glm::vec3*>(src),
        reinterpret_cast<float4*>(dst),
        numPixels
    );
}

void OptiXDenoiser::convertFloat4ToFloat3(CUdeviceptr src, void* dst, unsigned int numPixels)
{
    // Call the external CUDA kernel launcher
    launchConvertFloat4ToFloat3(
        reinterpret_cast<float4*>(src),
        reinterpret_cast<glm::vec3*>(dst),
        numPixels
    );
}

void OptiXDenoiser::denoise(void* beautyBuffer, void* normalBuffer, void* albedoBuffer, void* outputBuffer)
{
    if (!m_initialized) {
        std::cerr << "OptiX Denoiser not initialized!" << std::endl;
        return;
    }

    if (!m_enabled) {
        // If denoising is disabled, just copy input to output if needed
        if (outputBuffer && outputBuffer != beautyBuffer) {
            size_t pixelCount = m_imageWidth * m_imageHeight;
            CUDA_CHECK(cudaMemcpy(outputBuffer, beautyBuffer, pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
        }
        return;
    }

    size_t pixelCount = m_imageWidth * m_imageHeight;

    // Convert float3 beauty buffer to float4 for OptiX
    convertFloat3ToFloat4(beautyBuffer, m_beautyBuffer, pixelCount);

    // Convert normal buffer if provided
    if (m_useNormals && normalBuffer) {
        convertFloat3ToFloat4(normalBuffer, m_normalBuffer, pixelCount);
    }

    // Convert albedo buffer if provided
    if (m_useAlbedo && albedoBuffer) {
        convertFloat3ToFloat4(albedoBuffer, m_albedoBuffer, pixelCount);
    }

    // Setup input image
    OptixImage2D inputImage = {};
    inputImage.data = m_beautyBuffer;
    inputImage.width = m_imageWidth;
    inputImage.height = m_imageHeight;
    inputImage.rowStrideInBytes = m_imageWidth * sizeof(float4);
    inputImage.pixelStrideInBytes = sizeof(float4);
    inputImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Compute HDR intensity (required for HDR denoising)
    OPTIX_CHECK(optixDenoiserComputeIntensity(
        m_denoiser,
        0,  
        &inputImage,
        m_intensity,
        m_scratch,
        m_scratchSize
    ));

    // Compute average color
    OPTIX_CHECK(optixDenoiserComputeAverageColor(
        m_denoiser,
        0, 
        &inputImage,
        m_avgColor,
        m_scratch,
        m_scratchSize
    ));

    OptixDenoiserParams params = {};
    params.hdrIntensity = m_intensity;
    params.hdrAverageColor = m_avgColor;
    params.blendFactor = m_blendFactor;

    OptixDenoiserGuideLayer guideLayer = {};

    if (m_useAlbedo && albedoBuffer) {
        OptixImage2D albedoImage = {};
        albedoImage.data = m_albedoBuffer;
        albedoImage.width = m_imageWidth;
        albedoImage.height = m_imageHeight;
        albedoImage.rowStrideInBytes = m_imageWidth * sizeof(float4);
        albedoImage.pixelStrideInBytes = sizeof(float4);
        albedoImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        guideLayer.albedo = albedoImage;
    }

    if (m_useNormals && normalBuffer) {
        OptixImage2D normalImage = {};
        normalImage.data = m_normalBuffer;
        normalImage.width = m_imageWidth;
        normalImage.height = m_imageHeight;
        normalImage.rowStrideInBytes = m_imageWidth * sizeof(float4);
        normalImage.pixelStrideInBytes = sizeof(float4);
        normalImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        guideLayer.normal = normalImage;
    }

    // Setup output image
    OptixImage2D outputImage = {};
    outputImage.data = m_outputBuffer;
    outputImage.width = m_imageWidth;
    outputImage.height = m_imageHeight;
    outputImage.rowStrideInBytes = m_imageWidth * sizeof(float4);
    outputImage.pixelStrideInBytes = sizeof(float4);
    outputImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Setup layer
    OptixDenoiserLayer layer = {};
    layer.input = inputImage;
    layer.output = outputImage;

    // Execute denoiser
    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser,
        0,  // stream
        &params,
        m_state,
        m_stateSize,
        &guideLayer,
        &layer,
        1,  // num layers
        0,  // input offset X
        0,  // input offset Y
        m_scratch,
        m_scratchSize
    ));

    // Make sure denoising is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert float4 output back to float3
    if (outputBuffer) {
        convertFloat4ToFloat3(m_outputBuffer, outputBuffer, pixelCount);
    }
    else {
        // If no output buffer specified, write back to beauty buffer
        convertFloat4ToFloat3(m_outputBuffer, beautyBuffer, pixelCount);
    }
}

void OptiXDenoiser::cleanup()
{
    if (m_denoiser) {
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }

    if (m_context) {
        optixDeviceContextDestroy(m_context);
        m_context = nullptr;
    }

    // Free device memory
    if (m_scratch) cudaFree(reinterpret_cast<void*>(m_scratch));
    if (m_state) cudaFree(reinterpret_cast<void*>(m_state));
    if (m_intensity) cudaFree(reinterpret_cast<void*>(m_intensity));
    if (m_avgColor) cudaFree(reinterpret_cast<void*>(m_avgColor));
    if (m_beautyBuffer) cudaFree(reinterpret_cast<void*>(m_beautyBuffer));
    if (m_normalBuffer) cudaFree(reinterpret_cast<void*>(m_normalBuffer));
    if (m_albedoBuffer) cudaFree(reinterpret_cast<void*>(m_albedoBuffer));
    if (m_outputBuffer) cudaFree(reinterpret_cast<void*>(m_outputBuffer));

    m_scratch = 0;
    m_state = 0;
    m_intensity = 0;
    m_avgColor = 0;
    m_beautyBuffer = 0;
    m_normalBuffer = 0;
    m_albedoBuffer = 0;
    m_outputBuffer = 0;

    m_initialized = false;
}