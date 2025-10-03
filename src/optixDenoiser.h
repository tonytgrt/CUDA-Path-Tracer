#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>

// Forward declarations
struct Scene;

// Macro for checking OptiX errors
#define OPTIX_CHECK(call)                                                     \
    {                                                                          \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                           \
            std::cerr << "OptiX error in " << __FILE__ << " at line "        \
                      << __LINE__ << ": " << #call << " failed with error "  \
                      << res << std::endl;                                    \
            exit(1);                                                           \
        }                                                                      \
    }

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                      \
    {                                                                          \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << " at line "         \
                      << __LINE__ << ": " << #call << " failed with error "  \
                      << cudaGetErrorString(error) << std::endl;              \
            exit(1);                                                           \
        }                                                                      \
    }

class OptiXDenoiser
{
public:
    OptiXDenoiser();
    ~OptiXDenoiser();

    // Initialize the denoiser with image dimensions
    // useNormals: whether to use normal buffer as guide layer
    // useAlbedo: whether to use albedo buffer as guide layer
    void init(unsigned int width, unsigned int height, bool useNormals = true, bool useAlbedo = false);

    // Denoise the image
    // beautyBuffer: raw path traced image (device memory, float3/vec3 format)
    // normalBuffer: per-pixel normals (device memory, float3/vec3 format, optional)
    // albedoBuffer: per-pixel albedo (device memory, float3/vec3 format, optional)
    // outputBuffer: denoised result (device memory, float3/vec3 format)
    void denoise(void* beautyBuffer, void* normalBuffer = nullptr, void* albedoBuffer = nullptr, void* outputBuffer = nullptr);

    // Clean up resources
    void cleanup();

    // Check if denoiser is initialized
    bool isInitialized() const { return m_initialized; }

    // Set/get denoising blend factor (0 = full denoise, 1 = no denoise)
    void setBlendFactor(float factor) { m_blendFactor = factor; }
    float getBlendFactor() const { return m_blendFactor; }

    // Enable/disable denoising
    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool isEnabled() const { return m_enabled; }

private:
    // OptiX context and denoiser
    OptixDeviceContext m_context = nullptr;
    OptixDenoiser m_denoiser = nullptr;

    // Image dimensions
    unsigned int m_imageWidth = 0;
    unsigned int m_imageHeight = 0;

    // Denoiser options
    bool m_useNormals = true;
    bool m_useAlbedo = false;
    float m_blendFactor = 0.0f;  // 0 = full denoise, 1 = no denoise
    bool m_enabled = true;
    bool m_initialized = false;

    // Device memory for denoiser
    CUdeviceptr m_scratch = 0;
    CUdeviceptr m_state = 0;
    CUdeviceptr m_intensity = 0;
    CUdeviceptr m_avgColor = 0;

    // Allocated sizes
    size_t m_scratchSize = 0;
    size_t m_stateSize = 0;

    // Device buffers for image data (in float4 format for OptiX)
    CUdeviceptr m_beautyBuffer = 0;
    CUdeviceptr m_normalBuffer = 0;
    CUdeviceptr m_albedoBuffer = 0;
    CUdeviceptr m_outputBuffer = 0;

    // Convert float3 buffer to float4 (OptiX requirement)
    void convertFloat3ToFloat4(void* src, CUdeviceptr dst, unsigned int numPixels);

    // Convert float4 buffer back to float3
    void convertFloat4ToFloat3(CUdeviceptr src, void* dst, unsigned int numPixels);

    // Logging callback
    static void context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata);
};