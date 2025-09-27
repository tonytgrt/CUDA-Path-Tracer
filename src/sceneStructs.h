#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

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

    // Add mesh data for GLTF objects
    MeshData* meshData;  // Pointer to mesh data (nullptr for non-mesh objects)
    int triangleStart;   // Index into global triangle buffer
    int triangleCount;   // Number of triangles in this mesh
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
};

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

// New structure for environment map
struct EnvironmentMap
{
    glm::vec3* data;        // HDR pixel data (on GPU)
    int width;
    int height;
    float intensity;
    bool enabled;
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
};
