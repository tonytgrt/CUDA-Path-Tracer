#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define RR_START_BOUNCE 3        // Start Russian Roulette after this many bounces
#define RR_SURVIVAL_MIN 0.05f    // Minimum survival probability (5%)
#define RR_SURVIVAL_MAX 0.95f    // Maximum survival probability (95%)

enum GeomType
{
    SPHERE,
    CUBE,
    GLTF_MESH
};

enum MaterialType
{
    DIFFUSE,
    SPECULAR,
    REFRACTIVE,
    PBR,
    EMITTING
};

struct Texture
{
    unsigned char* data;
    int width;
    int height;
    int components; // 3 for RGB, 4 for RGBA
};

struct Triangle
{
    glm::vec3 v0, v1, v2;    // Vertices
    glm::vec3 n0, n1, n2;    // Normals (for smooth shading)
    glm::vec2 uv0, uv1, uv2; // Texture coordinates
    int materialId;
};

struct MeshData
{
    std::vector<Triangle> triangles;
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    MeshData* meshData;  // CPU Pointer to mesh data (nullptr for non-mesh objects)
    int triangleStart;   // Index into global triangle buffer
    int triangleCount;   // Number of triangles in this mesh

    int bvhIndex;       // Index into BVH arrays (-1 if no BVH)
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float transparency;
    float roughness;
	float metallic;
    float indexOfRefraction;
    float emittance;
	enum MaterialType type;

    // Add texture indices (-1 means no texture)
    int baseColorTextureIdx;
    int metallicRoughnessTextureIdx;
    int normalTextureIdx;
    int emissiveTextureIdx;
    int occlusionTextureIdx;

    // PBR factors (used when textures are not present)
    glm::vec3 emissiveFactor;

    // Subsurface scattering parameters
    glm::vec3 subsurfaceColor;        // Subsurface albedo (what color light becomes as it scatters)
    float subsurfaceRadius;            // Mean free path (average scatter distance)
    glm::vec3 subsurfaceRadiusRGB;    // Per-channel scattering radius for artistic control
    float subsurfaceScale;             // Global scale factor for the effect
    float subsurfaceAnisotropy;        // Phase function anisotropy (-1 to 1)
    int subsurfaceEnabled;             // 0 = disabled, 1 = enabled

    // Optional: additional SSS parameters for advanced control
    float subsurfaceIOR;               // Internal IOR for more accurate Fresnel
    float subsurfaceThickness;         // Thickness map value (for thin objects)
    glm::vec3 subsurfaceAbsorption;    // Absorption coefficient (alternative to color)

    // Padding for GPU alignment if needed
    float padding[1]; 
};

namespace SSSProfiles {
    // Skin profile based on d'Eon et al. 2007
    const glm::vec3 SKIN_RADIUS_RGB = glm::vec3(0.032f, 0.017f, 0.005f);
    const glm::vec3 SKIN_COLOR = glm::vec3(0.85f, 0.57f, 0.50f);

    // Milk/dairy products
    const glm::vec3 MILK_RADIUS_RGB = glm::vec3(0.10f, 0.08f, 0.05f);
    const glm::vec3 MILK_COLOR = glm::vec3(0.95f, 0.93f, 0.88f);

    // Marble
    const glm::vec3 MARBLE_RADIUS_RGB = glm::vec3(0.02f, 0.016f, 0.011f);
    const glm::vec3 MARBLE_COLOR = glm::vec3(0.95f, 0.88f, 0.82f);

    // Wax
    const glm::vec3 WAX_RADIUS_RGB = glm::vec3(0.08f, 0.06f, 0.03f);
    const glm::vec3 WAX_COLOR = glm::vec3(0.9f, 0.85f, 0.7f);

    // Jade
    const glm::vec3 JADE_RADIUS_RGB = glm::vec3(0.04f, 0.055f, 0.045f);
    const glm::vec3 JADE_COLOR = glm::vec3(0.45f, 0.70f, 0.55f);

    // Apple flesh
    const glm::vec3 APPLE_RADIUS_RGB = glm::vec3(0.03f, 0.02f, 0.01f);
    const glm::vec3 APPLE_COLOR = glm::vec3(0.92f, 0.90f, 0.75f);

    // Potato
    const glm::vec3 POTATO_RADIUS_RGB = glm::vec3(0.014f, 0.007f, 0.002f);
    const glm::vec3 POTATO_COLOR = glm::vec3(0.87f, 0.77f, 0.62f);

    // Ketchup
    const glm::vec3 KETCHUP_RADIUS_RGB = glm::vec3(0.061f, 0.027f, 0.002f);
    const glm::vec3 KETCHUP_COLOR = glm::vec3(0.92f, 0.12f, 0.05f);
}

struct SSSample {
    glm::vec3 position;      // Sample position relative to surface
    glm::vec3 irradiance;    // Cached irradiance at this sample
    float weight;            // Sample weight
    float radius;            // Distance from surface point
};

struct MultiLayerSSS {
    // Epidermis (outer layer)
    glm::vec3 epidermisColor;
    glm::vec3 epidermisRadius;
    float epidermisThickness;

    // Dermis (middle layer)
    glm::vec3 dermisColor;
    glm::vec3 dermisRadius;
    float dermisThickness;

    // Hypodermis (deep layer)
    glm::vec3 hypodermisColor;
    glm::vec3 hypodermisRadius;

    // Blend factors
    float layerBlend;
    float oiliness;  // Affects surface reflection
};

// ==========================================
// Helper function to initialize default SSS values
// ==========================================

inline void initializeDefaultSSS(Material& mat) {
    mat.subsurfaceEnabled = 0;
    mat.subsurfaceColor = glm::vec3(1.0f);
    mat.subsurfaceRadius = 0.01f;
    mat.subsurfaceRadiusRGB = glm::vec3(0.01f);
    mat.subsurfaceScale = 1.0f;
    mat.subsurfaceAnisotropy = 0.0f;
    mat.subsurfaceIOR = 1.4f;
    mat.subsurfaceThickness = 1.0f;
    mat.subsurfaceAbsorption = glm::vec3(0.0f);
}

struct LightInfo {
    int geomIdx;
    float area;
    float pdf; 
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct EnvMapDistribution {
    float* marginalCDF;      // CDF for selecting row (theta)
    float* conditionalCDFs;  // CDFs for selecting column within each row (phi)
    float totalPower;        // Total luminance of environment map
    int width;
    int height;
};

// New structure for environment map
struct EnvironmentMap
{
    glm::vec3* data;        // HDR pixel data (on GPU)
    int width;
    int height;
    float intensity;
    bool enabled;
    EnvMapDistribution distribution;
};

struct HostEnvironmentMap
{
    std::vector<glm::vec3> data;  // HDR pixel data (on CPU)
    int width;
    int height;
    float intensity;
    bool enabled;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
	bool prevIsSpecular = false;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};

struct GPUTexture
{
    unsigned char* data;  // Pointer to GPU memory
    int width;
    int height;
    int components;
};
