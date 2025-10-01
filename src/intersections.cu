#include "intersections.h"
#include "bvh.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    const Ray& r,
    glm::vec3& intersect,
    glm::vec3& barycentrics)
{
    const float EPSILON = 0.0000001f;

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);

    // Ray is parallel to triangle
    if (a > -EPSILON && a < EPSILON) {
        return -1.0f;
    }

    float f = 1.0f / a;
    glm::vec3 s = r.origin - v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }

    // Calculate t to find intersection point
    float t = f * glm::dot(edge2, q);

    if (t > EPSILON) {
        // Calculate barycentric coordinates
        float w = 1.0f - u - v;
        barycentrics = glm::vec3(w, u, v);  // (1-u-v, u, v) for (v0, v1, v2)

        // Calculate intersection point
        intersect = r.origin + r.direction * t;

        return t;
    }

    return -1.0f;
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    int& materialId)
{
    // Transform ray to object space
    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = FLT_MAX;
    glm::vec3 bestBarycentrics;
    int bestTriangleIdx = -1;
    glm::vec3 bestIntersect;

    // Test all triangles in the mesh
    for (int i = 0; i < mesh.triangleCount; i++) {
        int triIdx = mesh.triangleStart + i;
        const Triangle& tri = triangles[triIdx];

        glm::vec3 barycentrics;
        glm::vec3 intersect;

        float t = triangleIntersectionTest(
            tri.v0, tri.v1, tri.v2,
            localRay,
            intersect,
            barycentrics
        );

        if (t > 0.0f && t < t_min) {
            t_min = t;
            bestBarycentrics = barycentrics;
            bestTriangleIdx = triIdx;
            bestIntersect = intersect;
        }
    }

    if (bestTriangleIdx < 0) {
        return -1.0f;
    }

    // We found an intersection
    const Triangle& hitTri = triangles[bestTriangleIdx];

    // Interpolate normal using barycentric coordinates
    glm::vec3 localNormal = glm::normalize(
        bestBarycentrics.x * hitTri.n0 +
        bestBarycentrics.y * hitTri.n1 +
        bestBarycentrics.z * hitTri.n2
    );

    // Interpolate UV coordinates
    uv = bestBarycentrics.x * hitTri.uv0 +
        bestBarycentrics.y * hitTri.uv1 +
        bestBarycentrics.z * hitTri.uv2;

    // Get material ID from triangle
    materialId = hitTri.materialId;

    // Transform intersection point and normal back to world space
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(bestIntersect, 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(localNormal, 0.0f)));

    // Determine if ray is coming from outside
    // (dot product of ray direction and normal should be negative for outside hits)
    outside = glm::dot(r.direction, normal) < 0.0f;
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// Ray-AABB intersection helper
__host__ __device__ inline bool intersectAABB(
    const Ray& ray,
    const AABB& box,
    float& tMin,
    float& tMax)
{
    glm::vec3 invDir = 1.0f / ray.direction;
    glm::vec3 t0 = (box.min - ray.origin) * invDir;
    glm::vec3 t1 = (box.max - ray.origin) * invDir;

    glm::vec3 tmin = glm::min(t0, t1);
    glm::vec3 tmax = glm::max(t0, t1);

    tMin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    tMax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    return tMax >= tMin && tMax >= 0.0f;
}

// Triangle intersection helper for BVH
__host__ __device__ inline float triangleIntersectionBVH(
    const Triangle& tri,
    const Ray& ray,
    glm::vec3& barycentrics)
{
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

// Modified mesh intersection test with BVH support
__host__ __device__ float meshIntersectionTestBVH(
    const Geom& mesh,
    const Triangle* triangles,
    const BVHNode* bvhNodes,
    const int* bvhTriangleIndices,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    int& materialId)
{
    // Transform ray to object space
    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // Use BVH traversal if available
    if (bvhNodes != nullptr && bvhTriangleIndices != nullptr) {
        // Stack for iterative traversal
        int stack[BVH_TRAVERSAL_STACK_SIZE];
        int stackPtr = 0;

        // Start with root node
        stack[stackPtr++] = 0;

        float closestT = FLT_MAX;
        bool hitFound = false;
        glm::vec3 bestBarycentrics;
        int bestTriIdx = -1;

        const Triangle* meshTriangles = triangles + mesh.triangleStart;

        while (stackPtr > 0) {
            // Pop node from stack
            int nodeIdx = stack[--stackPtr];
            const BVHNode& node = bvhNodes[nodeIdx];

            // Check AABB intersection
            float tMin, tMax;
            if (!intersectAABB(localRay, node.bounds, tMin, tMax)) {
                continue;
            }

            // Early termination if this node is further than closest hit
            if (tMin > closestT) {
                continue;
            }

            if (node.isLeaf()) {
                // Test triangles in leaf
                for (int i = 0; i < node.triangleCount; i++) {
                    int triIdx = bvhTriangleIndices[node.triangleStart + i];
                    const Triangle& tri = meshTriangles[triIdx];

                    glm::vec3 barycentrics;
                    float triT = triangleIntersectionBVH(tri, localRay, barycentrics);

                    if (triT > 0.0f && triT < closestT) {
                        closestT = triT;
                        bestBarycentrics = barycentrics;
                        bestTriIdx = triIdx;
                        hitFound = true;
                    }
                }
            }
            else {
                // Add children to stack (traverse closer child first)
                if (node.leftChild >= 0 && stackPtr < BVH_TRAVERSAL_STACK_SIZE) {
                    stack[stackPtr++] = node.leftChild;
                }
                if (node.rightChild >= 0 && stackPtr < BVH_TRAVERSAL_STACK_SIZE) {
                    stack[stackPtr++] = node.rightChild;
                }
            }
        }

        if (hitFound) {
            const Triangle& hitTri = meshTriangles[bestTriIdx];

            // Interpolate normal
            glm::vec3 localNormal = glm::normalize(
                bestBarycentrics.x * hitTri.n0 +
                bestBarycentrics.y * hitTri.n1 +
                bestBarycentrics.z * hitTri.n2
            );

            // Interpolate UV
            uv = bestBarycentrics.x * hitTri.uv0 +
                bestBarycentrics.y * hitTri.uv1 +
                bestBarycentrics.z * hitTri.uv2;

            materialId = hitTri.materialId;

            // Calculate intersection point in object space
            glm::vec3 localIntersect = localRay.origin + localRay.direction * closestT;

            // Transform back to world space
            intersectionPoint = multiplyMV(mesh.transform, glm::vec4(localIntersect, 1.0f));
            normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(localNormal, 0.0f)));

            // Determine if ray is coming from outside
            outside = glm::dot(r.direction, normal) < 0.0f;
            if (!outside) {
                normal = -normal;
            }

            return glm::length(r.origin - intersectionPoint);
        }
        return -1.0f;
    }

    // Fallback to linear search if no BVH
    float t_min = FLT_MAX;
    glm::vec3 bestBarycentrics;
    int bestTriangleIdx = -1;
    glm::vec3 bestIntersect;

    // Test all triangles in the mesh
    for (int i = 0; i < mesh.triangleCount; i++) {
        int triIdx = mesh.triangleStart + i;
        const Triangle& tri = triangles[triIdx];

        glm::vec3 barycentrics;
        glm::vec3 intersect;

        float t = triangleIntersectionTest(
            tri.v0, tri.v1, tri.v2,
            localRay,
            intersect,
            barycentrics
        );

        if (t > 0.0f && t < t_min) {
            t_min = t;
            bestBarycentrics = barycentrics;
            bestTriangleIdx = triIdx;
            bestIntersect = intersect;
        }
    }

    if (bestTriangleIdx < 0) {
        return -1.0f;
    }

    // We found an intersection
    const Triangle& hitTri = triangles[bestTriangleIdx];

    // Interpolate normal using barycentric coordinates
    glm::vec3 localNormal = glm::normalize(
        bestBarycentrics.x * hitTri.n0 +
        bestBarycentrics.y * hitTri.n1 +
        bestBarycentrics.z * hitTri.n2
    );

    // Interpolate UV coordinates
    uv = bestBarycentrics.x * hitTri.uv0 +
        bestBarycentrics.y * hitTri.uv1 +
        bestBarycentrics.z * hitTri.uv2;

    // Get material ID from triangle
    materialId = hitTri.materialId;

    // Transform intersection point and normal back to world space
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(bestIntersect, 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(localNormal, 0.0f)));

    // Determine if ray is coming from outside
    outside = glm::dot(r.direction, normal) < 0.0f;
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}