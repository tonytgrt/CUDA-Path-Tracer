#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "glm/glm.hpp"
#include "sceneStructs.h"

// BVH Configuration
#define BVH_MAX_DEPTH 32
#define BVH_MAX_LEAF_TRIANGLES 4
#define BVH_TRAVERSAL_STACK_SIZE 64
#define BVH_SAH_SAMPLES 12

// AABB structure for bounding volumes
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    
    __host__ __device__ AABB() : 
        min(glm::vec3(FLT_MAX)), 
        max(glm::vec3(-FLT_MAX)) {}
    
    __host__ __device__ AABB(const glm::vec3& min, const glm::vec3& max) : 
        min(min), max(max) {}
    
    __host__ __device__ void expand(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
    
    __host__ __device__ void expand(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }
    
    __host__ __device__ glm::vec3 centroid() const {
        return (min + max) * 0.5f;
    }
    
    __host__ __device__ float surfaceArea() const {
        glm::vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    
    __host__ __device__ int longestAxis() const {
        glm::vec3 d = max - min;
        if (d.x > d.y && d.x > d.z) return 0;
        if (d.y > d.z) return 1;
        return 2;
    }
};

// GPU-friendly BVH node structure
struct BVHNode {
    AABB bounds;
    int leftChild;   // Index to left child, or -1 if leaf
    int rightChild;  // Index to right child
    int triangleStart; // Starting index in triangle array (for leaves)
    int triangleCount; // Number of triangles (for leaves)
    
    __host__ __device__ bool isLeaf() const {
        return leftChild == -1;
    }
};

// Triangle reference for BVH building
struct TriangleRef {
    int index;
    AABB bounds;
    glm::vec3 centroid;
};

// BVH build statistics
struct BVHStats {
    int nodeCount;
    int leafCount;
    int maxDepth;
    int totalTriangles;
    float buildTime;
    
    void print() const;
};

// Main BVH class
class BVH {
public:
    BVH();
    ~BVH();
    
    // Build BVH from triangles
    void build(const std::vector<Triangle>& triangles, int maxDepth = BVH_MAX_DEPTH);
    
    // Upload BVH to GPU
    void uploadToGPU();
    
    // Free GPU memory
    void freeGPU();
    
    // Get device pointers
    BVHNode* getDeviceNodes() const { return dev_nodes; }
    int* getDeviceTriangleIndices() const { return dev_triangleIndices; }
    int getNodeCount() const { return nodes.size(); }
    
    // Get statistics
    const BVHStats& getStats() const { return stats; }
    
private:
    // CPU data
    std::vector<BVHNode> nodes;
    std::vector<int> triangleIndices;
    std::vector<TriangleRef> triangleRefs;
    int maxDepth;
    
    // GPU data
    BVHNode* dev_nodes;
    int* dev_triangleIndices;
    
    // Statistics
    BVHStats stats;
    
    // Build functions
    int buildRecursive(int start, int end, int depth);
    int createLeafNode(int start, int end);
    int splitMiddle(std::vector<TriangleRef>& refs, int start, int end, int axis);
    int splitSAH(std::vector<TriangleRef>& refs, int start, int end);
    float evaluateSAH(const std::vector<TriangleRef>& refs, int start, int end, 
                      int axis, float splitPos, int& outSplit);
};

// GPU BVH traversal structure
struct BVHTraversal {
    BVHNode* nodes;
    int* triangleIndices;
    Triangle* triangles;
    int maxStackSize;
    
    __device__ BVHTraversal(BVHNode* n, int* ti, Triangle* t, int stackSize = BVH_TRAVERSAL_STACK_SIZE) :
        nodes(n), triangleIndices(ti), triangles(t), maxStackSize(stackSize) {}
    
    // Ray-AABB intersection test
    __device__ bool intersectAABB(const Ray& ray, const AABB& box, float& tMin, float& tMax) const;
    
    // Traverse BVH and find closest triangle intersection
    __device__ bool traverse(const Ray& ray, float& t, glm::vec3& normal, 
                            glm::vec2& uv, int& triangleIdx, int& materialId) const;
};

// Helper functions for GPU
__device__ inline bool rayAABBIntersection(
    const Ray& ray, 
    const AABB& box,
    float& tMin,
    float& tMax
);

__device__ inline float triangleIntersectionBVH(
    const Triangle& tri,
    const Ray& ray,
    glm::vec3& barycentrics
);