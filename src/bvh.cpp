#include "bvh.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <stack>
#include <cuda_runtime.h>

// BVH Stats implementation
void BVHStats::print() const {
    std::cout << "=== BVH Statistics ===" << std::endl;
    std::cout << "Total Nodes: " << nodeCount << std::endl;
    std::cout << "Leaf Nodes: " << leafCount << std::endl;
    std::cout << "Max Depth: " << maxDepth << std::endl;
    std::cout << "Total Triangles: " << totalTriangles << std::endl;
    std::cout << "Build Time: " << buildTime << " ms" << std::endl;
    std::cout << "Avg Triangles/Leaf: " << (float)totalTriangles / leafCount << std::endl;
}

// BVH Constructor
BVH::BVH() : dev_nodes(nullptr), dev_triangleIndices(nullptr), maxDepth(BVH_MAX_DEPTH) {
    stats = {};
}

// BVH Destructor
BVH::~BVH() {
    freeGPU();
}

// Build BVH from triangles
void BVH::build(const std::vector<Triangle>& triangles, int maxDepthParam) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    maxDepth = maxDepthParam;
    nodes.clear();
    triangleIndices.clear();
    triangleRefs.clear();
    
    // Create triangle references with bounds
    triangleRefs.reserve(triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
        TriangleRef ref;
        ref.index = i;
        ref.bounds = AABB();
        ref.bounds.expand(triangles[i].v0);
        ref.bounds.expand(triangles[i].v1);
        ref.bounds.expand(triangles[i].v2);
        ref.centroid = ref.bounds.centroid();
        triangleRefs.push_back(ref);
    }
    
    // Reserve space for nodes (rough estimate)
    nodes.reserve(2 * triangles.size());
    triangleIndices.reserve(triangles.size());
    
    // Build tree recursively
    stats = {};
    stats.totalTriangles = triangles.size();
    buildRecursive(0, triangleRefs.size(), 0);
    
    // Compute stats
    stats.nodeCount = nodes.size();
    for (const auto& node : nodes) {
        if (node.isLeaf()) stats.leafCount++;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    stats.buildTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    
    std::cout << "BVH built with " << nodes.size() << " nodes for " 
              << triangles.size() << " triangles in " << stats.buildTime << " ms" << std::endl;
}

// Recursive BVH building
int BVH::buildRecursive(int start, int end, int depth) {
    int nodeIdx = nodes.size();
    nodes.push_back(BVHNode());
    BVHNode& node = nodes[nodeIdx];
    
    // Update max depth stat
    stats.maxDepth = std::max(stats.maxDepth, depth);
    
    // Compute bounds for this node
    node.bounds = AABB();
    for (int i = start; i < end; i++) {
        node.bounds.expand(triangleRefs[i].bounds);
    }
    
    int count = end - start;
    
    // Create leaf if we've reached max depth or have few triangles
    if (depth >= maxDepth || count <= BVH_MAX_LEAF_TRIANGLES) {
        return createLeafNode(start, end);
    }
    
    // Choose split strategy
    int mid;
    if (count > 32) {
        // Use SAH for larger nodes
        mid = splitSAH(triangleRefs, start, end);
    } else {
        // Use simple median split for smaller nodes
        int axis = node.bounds.longestAxis();
        mid = splitMiddle(triangleRefs, start, end, axis);
    }
    
    // Ensure valid split
    if (mid == start || mid == end) {
        return createLeafNode(start, end);
    }
    
    // Recursively build children
    node.leftChild = buildRecursive(start, mid, depth + 1);
    node.rightChild = buildRecursive(mid, end, depth + 1);
    node.triangleStart = -1;
    node.triangleCount = 0;
    
    return nodeIdx;
}

// Create leaf node
int BVH::createLeafNode(int start, int end) {
    int nodeIdx = nodes.size() - 1;  // We already pushed the node
    BVHNode& node = nodes[nodeIdx];
    
    node.leftChild = -1;
    node.rightChild = -1;
    node.triangleStart = triangleIndices.size();
    node.triangleCount = end - start;
    
    // Add triangle indices
    for (int i = start; i < end; i++) {
        triangleIndices.push_back(triangleRefs[i].index);
    }
    
    return nodeIdx;
}

// Split using median along an axis
int BVH::splitMiddle(std::vector<TriangleRef>& refs, int start, int end, int axis) {
    int mid = (start + end) / 2;
    
    // Sort by centroid along chosen axis
    std::nth_element(refs.begin() + start, refs.begin() + mid, refs.begin() + end,
        [axis](const TriangleRef& a, const TriangleRef& b) {
            return a.centroid[axis] < b.centroid[axis];
        });
    
    return mid;
}

// Split using Surface Area Heuristic
int BVH::splitSAH(std::vector<TriangleRef>& refs, int start, int end) {
    if (end - start <= 2) {
        return (start + end) / 2;
    }
    
    AABB bounds;
    for (int i = start; i < end; i++) {
        bounds.expand(refs[i].centroid);
    }
    
    int bestAxis = -1;
    float bestPos = 0;
    float bestCost = FLT_MAX;
    int bestSplit = -1;
    
    // Try splitting along each axis
    for (int axis = 0; axis < 3; axis++) {
        float boundsMin = bounds.min[axis];
        float boundsMax = bounds.max[axis];
        
        if (boundsMin == boundsMax) continue;
        
        // Sample split positions
        for (int i = 1; i < BVH_SAH_SAMPLES; i++) {
            float t = i / (float)BVH_SAH_SAMPLES;
            float splitPos = boundsMin + t * (boundsMax - boundsMin);
            
            int split;
            float cost = evaluateSAH(refs, start, end, axis, splitPos, split);
            
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestPos = splitPos;
                bestSplit = split;
            }
        }
    }
    
    // If no good split found, fall back to median
    if (bestAxis == -1 || bestSplit <= start || bestSplit >= end) {
        int axis = bounds.longestAxis();
        return splitMiddle(refs, start, end, axis);
    }
    
    // Partition based on best split
    auto pivot = std::partition(refs.begin() + start, refs.begin() + end,
        [bestAxis, bestPos](const TriangleRef& ref) {
            return ref.centroid[bestAxis] < bestPos;
        });
    
    return pivot - refs.begin();
}

// Evaluate SAH cost for a split
float BVH::evaluateSAH(const std::vector<TriangleRef>& refs, int start, int end,
                      int axis, float splitPos, int& outSplit) {
    AABB leftBounds, rightBounds;
    int leftCount = 0, rightCount = 0;
    
    for (int i = start; i < end; i++) {
        if (refs[i].centroid[axis] < splitPos) {
            leftBounds.expand(refs[i].bounds);
            leftCount++;
        } else {
            rightBounds.expand(refs[i].bounds);
            rightCount++;
        }
    }
    
    outSplit = start + leftCount;
    
    if (leftCount == 0 || rightCount == 0) {
        return FLT_MAX;
    }
    
    float leftArea = leftBounds.surfaceArea();
    float rightArea = rightBounds.surfaceArea();
    float totalArea = leftArea + rightArea;
    
    // SAH cost: traversal_cost + (leftArea/totalArea) * leftCount + (rightArea/totalArea) * rightCount
    const float traversalCost = 1.0f;
    const float intersectionCost = 1.0f;
    
    return traversalCost + intersectionCost * 
           (leftArea / totalArea * leftCount + rightArea / totalArea * rightCount);
}

// Upload BVH to GPU
void BVH::uploadToGPU() {
    freeGPU();
    
    if (nodes.empty()) return;
    
    // Allocate and copy nodes
    size_t nodesSize = nodes.size() * sizeof(BVHNode);
    cudaMalloc(&dev_nodes, nodesSize);
    cudaMemcpy(dev_nodes, nodes.data(), nodesSize, cudaMemcpyHostToDevice);
    
    // Allocate and copy triangle indices
    if (!triangleIndices.empty()) {
        size_t indicesSize = triangleIndices.size() * sizeof(int);
        cudaMalloc(&dev_triangleIndices, indicesSize);
        cudaMemcpy(dev_triangleIndices, triangleIndices.data(), indicesSize, cudaMemcpyHostToDevice);
    }
    
    std::cout << "BVH uploaded to GPU: " << nodesSize / 1024.0f << " KB for nodes, "
              << triangleIndices.size() * sizeof(int) / 1024.0f << " KB for indices" << std::endl;
}

// Free GPU memory
void BVH::freeGPU() {
    if (dev_nodes) {
        cudaFree(dev_nodes);
        dev_nodes = nullptr;
    }
    if (dev_triangleIndices) {
        cudaFree(dev_triangleIndices);
        dev_triangleIndices = nullptr;
    }
}

// GPU traversal implementation
__device__ bool BVHTraversal::intersectAABB(const Ray& ray, const AABB& box, float& tMin, float& tMax) const {
    glm::vec3 invDir = 1.0f / ray.direction;
    glm::vec3 t0 = (box.min - ray.origin) * invDir;
    glm::vec3 t1 = (box.max - ray.origin) * invDir;
    
    glm::vec3 tmin = glm::min(t0, t1);
    glm::vec3 tmax = glm::max(t0, t1);
    
    tMin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    tMax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);
    
    return tMax >= tMin && tMax >= 0.0f;
}

__device__ bool BVHTraversal::traverse(const Ray& ray, float& t, glm::vec3& normal, 
                                       glm::vec2& uv, int& triangleIdx, int& materialId) const {
    // Stack for iterative traversal
    int stack[BVH_TRAVERSAL_STACK_SIZE];
    int stackPtr = 0;
    
    // Start with root node
    stack[stackPtr++] = 0;
    
    float closestT = FLT_MAX;
    bool hitFound = false;
    glm::vec3 bestBarycentrics;
    int bestTriIdx = -1;
    
    while (stackPtr > 0) {
        // Pop node from stack
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = nodes[nodeIdx];
        
        // Check AABB intersection
        float tMin, tMax;
        if (!intersectAABB(ray, node.bounds, tMin, tMax)) {
            continue;
        }
        
        // Early termination if this node is further than closest hit
        if (tMin > closestT) {
            continue;
        }
        
        if (node.isLeaf()) {
            // Test triangles in leaf
            for (int i = 0; i < node.triangleCount; i++) {
                int triIdx = triangleIndices[node.triangleStart + i];
                const Triangle& tri = triangles[triIdx];
                
                glm::vec3 barycentrics;
                float triT = triangleIntersectionBVH(tri, ray, barycentrics);
                
                if (triT > 0.0f && triT < closestT) {
                    closestT = triT;
                    bestBarycentrics = barycentrics;
                    bestTriIdx = triIdx;
                    hitFound = true;
                }
            }
        } else {
            // Add children to stack (traverse closer child first)
            if (node.leftChild >= 0 && stackPtr < maxStackSize) {
                stack[stackPtr++] = node.leftChild;
            }
            if (node.rightChild >= 0 && stackPtr < maxStackSize) {
                stack[stackPtr++] = node.rightChild;
            }
        }
    }
    
    if (hitFound) {
        const Triangle& hitTri = triangles[bestTriIdx];
        
        // Interpolate normal
        normal = glm::normalize(
            bestBarycentrics.x * hitTri.n0 +
            bestBarycentrics.y * hitTri.n1 +
            bestBarycentrics.z * hitTri.n2
        );
        
        // Interpolate UV
        uv = bestBarycentrics.x * hitTri.uv0 +
             bestBarycentrics.y * hitTri.uv1 +
             bestBarycentrics.z * hitTri.uv2;
        
        triangleIdx = bestTriIdx;
        materialId = hitTri.materialId;
        t = closestT;
        
        return true;
    }
    
    return false;
}

// Helper GPU functions
__device__ inline bool rayAABBIntersection(const Ray& ray, const AABB& box, 
                                          float& tMin, float& tMax) {
    glm::vec3 invDir = 1.0f / ray.direction;
    glm::vec3 t0 = (box.min - ray.origin) * invDir;
    glm::vec3 t1 = (box.max - ray.origin) * invDir;
    
    glm::vec3 tmin = glm::min(t0, t1);
    glm::vec3 tmax = glm::max(t0, t1);
    
    tMin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    tMax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);
    
    return tMax >= tMin && tMax >= 0.0f;
}

__device__ inline float triangleIntersectionBVH(const Triangle& tri, const Ray& ray, 
                                               glm::vec3& barycentrics) {
    const float EPSILON = 0.0000001f;
    
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) {
        return -1.0f;
    }
    
    float f = 1.0f / a;
    glm::vec3 s = ray.origin - tri.v0;
    float u = f * glm::dot(s, h);
    
    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }
    
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }
    
    float t = f * glm::dot(edge2, q);
    
    if (t > EPSILON) {
        float w = 1.0f - u - v;
        barycentrics = glm::vec3(w, u, v);
        return t;
    }
    
    return -1.0f;
}