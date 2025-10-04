#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/quaternion.hpp>       
#include <glm/gtx/quaternion.hpp>        
#include <glm/gtc/matrix_transform.hpp>

#include "json.hpp"

#include "tiny_gltf.h"
#include "stb_image.h"


#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>


using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

// Load texture from glTF - now returns existing texture if already loaded
int Scene::loadGLTFTexture(const tinygltf::Texture& gltfTex,
    const tinygltf::Model& model,
    int gltfTextureIndex) {

    // Check if this GLTF texture has already been loaded
    auto it = gltfTextureToSceneTexture.find(gltfTextureIndex);
    if (it != gltfTextureToSceneTexture.end()) {
        cout << "Reusing already loaded texture: GLTF index " << gltfTextureIndex
            << " -> Scene index " << it->second << endl;
        return it->second;
    }

    if (gltfTex.source < 0 || gltfTex.source >= model.images.size()) {
        cout << "Invalid texture source index: " << gltfTex.source << endl;
        return -1;
    }

    const tinygltf::Image& image = model.images[gltfTex.source];

    // Check if image has valid data
    if (image.image.empty()) {
        cout << "Warning: Empty image data for texture " << gltfTextureIndex << endl;
        return -1;
    }

    Texture tex;
    tex.width = image.width;
    tex.height = image.height;
    tex.components = image.component;

    // Allocate and copy image data
    size_t imageSize = tex.width * tex.height * tex.components;
    tex.data = new unsigned char[imageSize];
    memcpy(tex.data, image.image.data(), imageSize);

    textures.push_back(tex);
    int sceneTextureIdx = textures.size() - 1;

    // Store the mapping
    gltfTextureToSceneTexture[gltfTextureIndex] = sceneTextureIdx;

    cout << "Loaded texture: " << image.name
        << " (" << tex.width << "x" << tex.height << ", " << tex.components << " channels)"
        << " GLTF index " << gltfTextureIndex << " -> Scene index " << sceneTextureIdx << endl;

    return sceneTextureIdx;
}

// Load material from glTF
int Scene::loadGLTFMaterial(const tinygltf::Material& gltfMat,
    const tinygltf::Model& model,
    const std::unordered_map<int, int>& textureMapping) {
    Material mat;
    mat.type = PBR;

    // Initialize defaults
    mat.baseColorTextureIdx = -1;
    mat.metallicRoughnessTextureIdx = -1;
    mat.normalTextureIdx = -1;
    mat.emissiveTextureIdx = -1;
    mat.occlusionTextureIdx = -1;
    mat.transparency = 0.0f;
    mat.emittance = 0.0f;

    // Get PBR metallic roughness properties
    const auto& pbr = gltfMat.pbrMetallicRoughness;

    // Base color factor (used when texture is not present or as a multiplier)
    if (pbr.baseColorFactor.size() >= 3) {
        mat.color = glm::vec3(
            pbr.baseColorFactor[0],
            pbr.baseColorFactor[1],
            pbr.baseColorFactor[2]
        );
        if (pbr.baseColorFactor.size() >= 4) {
            mat.transparency = 1.0f - pbr.baseColorFactor[3]; // Alpha to transparency
        }
    }
    else {
        mat.color = glm::vec3(1.0f);
    }

    // Metallic and roughness factors
    mat.metallic = pbr.metallicFactor;
    mat.roughness = pbr.roughnessFactor;

    // Load textures using the pre-loaded mapping
    if (pbr.baseColorTexture.index >= 0) {
        auto it = textureMapping.find(pbr.baseColorTexture.index);
        if (it != textureMapping.end()) {
            mat.baseColorTextureIdx = it->second;
            cout << "Material uses base color texture: Scene index " << it->second << endl;
        }
    }

    if (pbr.metallicRoughnessTexture.index >= 0) {
        auto it = textureMapping.find(pbr.metallicRoughnessTexture.index);
        if (it != textureMapping.end()) {
            mat.metallicRoughnessTextureIdx = it->second;
            cout << "Material uses metallic-roughness texture: Scene index " << it->second << endl;
        }
    }

    if (gltfMat.normalTexture.index >= 0) {
        auto it = textureMapping.find(gltfMat.normalTexture.index);
        if (it != textureMapping.end()) {
            mat.normalTextureIdx = it->second;
            cout << "Material uses normal texture: Scene index " << it->second << endl;
        }
    }

    if (gltfMat.emissiveTexture.index >= 0) {
        auto it = textureMapping.find(gltfMat.emissiveTexture.index);
        if (it != textureMapping.end()) {
            mat.emissiveTextureIdx = it->second;
            cout << "Material uses emissive texture: Scene index " << it->second << endl;
        }
    }

    if (gltfMat.occlusionTexture.index >= 0) {
        auto it = textureMapping.find(gltfMat.occlusionTexture.index);
        if (it != textureMapping.end()) {
            mat.occlusionTextureIdx = it->second;
            cout << "Material uses occlusion texture: Scene index " << it->second << endl;
        }
    }

    // Load emissive factor
    if (gltfMat.emissiveFactor.size() >= 3) {
        mat.emissiveFactor = glm::vec3(
            gltfMat.emissiveFactor[0],
            gltfMat.emissiveFactor[1],
            gltfMat.emissiveFactor[2]
        );
        if (glm::length(mat.emissiveFactor) > 0.0f) {
            mat.emittance = glm::length(mat.emissiveFactor);
            mat.type = EMITTING;
        }
    }

    // Handle alpha mode
    if (gltfMat.alphaMode == "BLEND" || gltfMat.alphaMode == "MASK") {
        if (gltfMat.alphaMode == "MASK") {
            mat.transparency = (gltfMat.alphaCutoff > 0.5f) ? 1.0f : 0.0f;
        }
    }

    materials.push_back(mat);
    return materials.size() - 1;
}

void Scene::loadGLTFModel(const std::string& gltfPath, Geom& newGeom,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale) {

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;

    bool ret = false;

    // Check if it's a binary glTF file (.glb)
    if (gltfPath.find(".glb") != std::string::npos) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfPath);
    }
    else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    }

    if (!warn.empty()) {
        cout << "GLTF Warning: " << warn << endl;
    }

    if (!err.empty()) {
        cout << "GLTF Error: " << err << endl;
        return;
    }

    if (!ret) {
        cout << "Failed to load glTF: " << gltfPath << endl;
        return;
    }

    cout << "\n=== Loading GLTF Model ===" << endl;
    cout << "Textures in model: " << model.textures.size() << endl;
    cout << "Images in model: " << model.images.size() << endl;
    cout << "Materials in model: " << model.materials.size() << endl;

    // Clear the texture mapping for this model
    gltfTextureToSceneTexture.clear();

    std::unordered_map<int, int> localTextureMapping;
    cout << "\n--- Loading Textures ---" << endl;
    for (size_t i = 0; i < model.textures.size(); ++i) {
        int sceneTextureIdx = loadGLTFTexture(model.textures[i], model, i);
        localTextureMapping[i] = sceneTextureIdx;
    }
    cout << "Total textures loaded: " << localTextureMapping.size() << endl;

    int materialOffset = materials.size();
    std::vector<int> gltfMaterialToSceneMaterial;

    cout << "\n--- Loading Materials ---" << endl;
    for (const auto& gltfMat : model.materials) {
        int sceneMaterialIdx = loadGLTFMaterial(gltfMat, model, localTextureMapping);
        gltfMaterialToSceneMaterial.push_back(sceneMaterialIdx);
        cout << "Loaded material '" << gltfMat.name << "' -> Scene index " << sceneMaterialIdx << endl;
    }

    // If no materials in GLTF, create a default mapping
    if (gltfMaterialToSceneMaterial.empty()) {
        gltfMaterialToSceneMaterial.push_back(0);
        cout << "No materials in GLTF, using default material" << endl;
    }

    cout << "\nMaterial mapping summary:" << endl;
    cout << "Material offset: " << materialOffset << endl;
    cout << "Total materials in scene: " << materials.size() << endl;

    // Build the transformation matrix from the JSON parameters
    glm::mat4 jsonTransform = utilityCore::buildTransformationMatrix(
        translation, rotation, scale);

    // Create a new MeshData to store all triangles
    MeshData meshData;
    meshData.boundingBoxMin = glm::vec3(FLT_MAX);
    meshData.boundingBoxMax = glm::vec3(-FLT_MAX);

    // Process all nodes to get meshes
    cout << "\n--- Processing Mesh Geometry ---" << endl;
    for (const auto& scene : model.scenes) {
        for (int nodeIdx : scene.nodes) {
            std::function<void(int, glm::mat4)> processNode = [&](int nodeIdx, glm::mat4 parentTransform) {
                if (nodeIdx < 0 || nodeIdx >= model.nodes.size()) return;

                const tinygltf::Node& node = model.nodes[nodeIdx];

                // Calculate node transform
                glm::mat4 nodeTransform = parentTransform;

                // Apply node's own transformation if it has one
                if (node.matrix.size() == 16) {
                    glm::mat4 localTransform;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            localTransform[j][i] = static_cast<float>(node.matrix[i * 4 + j]);
                        }
                    }
                    nodeTransform = parentTransform * localTransform;
                }
                // Or use TRS (Translation, Rotation, Scale) if present
                else {
                    glm::mat4 T = glm::mat4(1.0f);
                    glm::mat4 R = glm::mat4(1.0f);
                    glm::mat4 S = glm::mat4(1.0f);

                    // Translation
                    if (node.translation.size() == 3) {
                        T = glm::translate(glm::mat4(1.0f),
                            glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
                    }

                    // Rotation (stored as quaternion)
                    if (node.rotation.size() == 4) {
                        glm::quat q(
                            static_cast<float>(node.rotation[3]), // w
                            static_cast<float>(node.rotation[0]), // x
                            static_cast<float>(node.rotation[1]), // y
                            static_cast<float>(node.rotation[2])  // z
                        );
                        R = glm::mat4_cast(q);
                    }

                    // Scale
                    if (node.scale.size() == 3) {
                        S = glm::scale(glm::mat4(1.0f),
                            glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
                    }

                    glm::mat4 localTransform = T * R * S;
                    nodeTransform = parentTransform * localTransform;
                }

                // Process mesh if present
                if (node.mesh >= 0 && node.mesh < model.meshes.size()) {
                    const tinygltf::Mesh& mesh = model.meshes[node.mesh];
                    cout << "Processing mesh: " << mesh.name << " with " << mesh.primitives.size() << " primitives" << endl;

                    for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
                        const auto& primitive = mesh.primitives[primIdx];

                        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                            cout << "Warning: Non-triangle primitive skipped" << endl;
                            continue;
                        }

                        // Check if POSITION attribute exists
                        if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                            cout << "Warning: Primitive without POSITION attribute skipped" << endl;
                            continue;
                        }

                        // Determine the correct material ID for this primitive
                        int sceneMaterialId = 0;
                        if (primitive.material >= 0 && primitive.material < gltfMaterialToSceneMaterial.size()) {
                            sceneMaterialId = gltfMaterialToSceneMaterial[primitive.material];
                        }
                        else if (!gltfMaterialToSceneMaterial.empty()) {
                            sceneMaterialId = gltfMaterialToSceneMaterial[0];
                        }

                        cout << "  Primitive " << primIdx << ": GLTF material " << primitive.material
                            << " -> Scene material " << sceneMaterialId << endl;

                        // Get accessor for positions
                        const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
                        const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
                        const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

                        const float* positions = reinterpret_cast<const float*>(
                            &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

                        // Get normals if available
                        const float* normals = nullptr;
                        if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                            const tinygltf::Accessor& normAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                            const tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
                            const tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];
                            normals = reinterpret_cast<const float*>(
                                &normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
                        }

                        // Get texture coordinates if available
                        const float* texcoords = nullptr;
                        if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                            const tinygltf::Accessor& texAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                            const tinygltf::BufferView& texBufferView = model.bufferViews[texAccessor.bufferView];
                            const tinygltf::Buffer& texBuffer = model.buffers[texBufferView.buffer];
                            texcoords = reinterpret_cast<const float*>(
                                &texBuffer.data[texBufferView.byteOffset + texAccessor.byteOffset]);
                        }

                        // Process triangles
                        if (primitive.indices >= 0) {
                            // Indexed geometry
                            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                            for (size_t i = 0; i < indexAccessor.count; i += 3) {
                                Triangle tri;

                                // Get vertex indices
                                unsigned int idx[3];
                                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                                    const unsigned short* indices = reinterpret_cast<const unsigned short*>(
                                        &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                                    idx[0] = indices[i];
                                    idx[1] = indices[i + 1];
                                    idx[2] = indices[i + 2];
                                }
                                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                                    const unsigned int* indices = reinterpret_cast<const unsigned int*>(
                                        &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                                    idx[0] = indices[i];
                                    idx[1] = indices[i + 1];
                                    idx[2] = indices[i + 2];
                                }
                                else {
                                    cout << "Unsupported index type" << endl;
                                    continue;
                                }

                                // Get vertex positions in local space
                                glm::vec3 localV0 = glm::vec3(positions[idx[0] * 3], positions[idx[0] * 3 + 1], positions[idx[0] * 3 + 2]);
                                glm::vec3 localV1 = glm::vec3(positions[idx[1] * 3], positions[idx[1] * 3 + 1], positions[idx[1] * 3 + 2]);
                                glm::vec3 localV2 = glm::vec3(positions[idx[2] * 3], positions[idx[2] * 3 + 1], positions[idx[2] * 3 + 2]);

                                tri.v0 = glm::vec3(nodeTransform * glm::vec4(localV0, 1.0f));
                                tri.v1 = glm::vec3(nodeTransform * glm::vec4(localV1, 1.0f));
                                tri.v2 = glm::vec3(nodeTransform * glm::vec4(localV2, 1.0f));

                                // Transform normals (use inverse transpose for correct normal transformation)
                                if (normals) {
                                    glm::vec3 localN0 = glm::vec3(normals[idx[0] * 3], normals[idx[0] * 3 + 1], normals[idx[0] * 3 + 2]);
                                    glm::vec3 localN1 = glm::vec3(normals[idx[1] * 3], normals[idx[1] * 3 + 1], normals[idx[1] * 3 + 2]);
                                    glm::vec3 localN2 = glm::vec3(normals[idx[2] * 3], normals[idx[2] * 3 + 1], normals[idx[2] * 3 + 2]);

                                    glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(nodeTransform)));
                                    tri.n0 = glm::normalize(normalMatrix * localN0);
                                    tri.n1 = glm::normalize(normalMatrix * localN1);
                                    tri.n2 = glm::normalize(normalMatrix * localN2);
                                }
                                else {
                                    // Calculate face normal from transformed vertices
                                    glm::vec3 faceNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                                    tri.n0 = tri.n1 = tri.n2 = faceNormal;
                                }

                                // Texture coordinates don't need transformation
                                if (texcoords) {
                                    tri.uv0 = glm::vec2(texcoords[idx[0] * 2], texcoords[idx[0] * 2 + 1]);
                                    tri.uv1 = glm::vec2(texcoords[idx[1] * 2], texcoords[idx[1] * 2 + 1]);
                                    tri.uv2 = glm::vec2(texcoords[idx[2] * 2], texcoords[idx[2] * 2 + 1]);
                                }
                                else {
                                    tri.uv0 = tri.uv1 = tri.uv2 = glm::vec2(0.0f);
                                }

                                tri.materialId = sceneMaterialId;

                                // Update bounding box with transformed vertices
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v0);
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v1);
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v2);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v0);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v1);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v2);

                                meshData.triangles.push_back(tri);
                            }
                        }
                        else {
                            // Non-indexed geometry - create triangles from vertex array
                            for (size_t i = 0; i < posAccessor.count; i += 3) {
                                Triangle tri;

                                // Get vertex positions in local space
                                glm::vec3 localV0 = glm::vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                                glm::vec3 localV1 = glm::vec3(positions[(i + 1) * 3], positions[(i + 1) * 3 + 1], positions[(i + 1) * 3 + 2]);
                                glm::vec3 localV2 = glm::vec3(positions[(i + 2) * 3], positions[(i + 2) * 3 + 1], positions[(i + 2) * 3 + 2]);

                                // Transform vertices by the combined node transform
                                tri.v0 = glm::vec3(nodeTransform * glm::vec4(localV0, 1.0f));
                                tri.v1 = glm::vec3(nodeTransform * glm::vec4(localV1, 1.0f));
                                tri.v2 = glm::vec3(nodeTransform * glm::vec4(localV2, 1.0f));

                                // Transform normals
                                if (normals) {
                                    glm::vec3 localN0 = glm::vec3(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
                                    glm::vec3 localN1 = glm::vec3(normals[(i + 1) * 3], normals[(i + 1) * 3 + 1], normals[(i + 1) * 3 + 2]);
                                    glm::vec3 localN2 = glm::vec3(normals[(i + 2) * 3], normals[(i + 2) * 3 + 1], normals[(i + 2) * 3 + 2]);

                                    glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(nodeTransform)));
                                    tri.n0 = glm::normalize(normalMatrix * localN0);
                                    tri.n1 = glm::normalize(normalMatrix * localN1);
                                    tri.n2 = glm::normalize(normalMatrix * localN2);
                                }
                                else {
                                    glm::vec3 faceNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                                    tri.n0 = tri.n1 = tri.n2 = faceNormal;
                                }

                                // Set texture coordinates
                                if (texcoords) {
                                    tri.uv0 = glm::vec2(texcoords[i * 2], texcoords[i * 2 + 1]);
                                    tri.uv1 = glm::vec2(texcoords[(i + 1) * 2], texcoords[(i + 1) * 2 + 1]);
                                    tri.uv2 = glm::vec2(texcoords[(i + 2) * 2], texcoords[(i + 2) * 2 + 1]);
                                }
                                else {
                                    tri.uv0 = tri.uv1 = tri.uv2 = glm::vec2(0.0f);
                                }

                                tri.materialId = sceneMaterialId;

                                // Update bounding box
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v0);
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v1);
                                meshData.boundingBoxMin = glm::min(meshData.boundingBoxMin, tri.v2);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v0);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v1);
                                meshData.boundingBoxMax = glm::max(meshData.boundingBoxMax, tri.v2);

                                meshData.triangles.push_back(tri);
                            }
                        }
                    }
                }

                // Process children with accumulated transform
                for (int childIdx : node.children) {
                    processNode(childIdx, nodeTransform);
                }
                };

            // START with the JSON transformation as the root transform
            processNode(nodeIdx, jsonTransform);
        }
    }

    // Store mesh data
    meshes.push_back(meshData);
    newGeom.meshData = &meshes.back();
    newGeom.triangleCount = meshData.triangles.size();

    cout << "Loaded GLTF model with " << newGeom.triangleCount << " triangles" << endl;
    cout << "Bounding box: (" << meshData.boundingBoxMin.x << "," << meshData.boundingBoxMin.y << "," << meshData.boundingBoxMin.z << ") to ("
        << meshData.boundingBoxMax.x << "," << meshData.boundingBoxMax.y << "," << meshData.boundingBoxMax.z << ")" << endl;

    // Debug: Print material usage summary
    if (meshData.triangles.size() > 0) {
        std::unordered_map<int, int> materialCounts;
        for (const auto& tri : meshData.triangles) {
            materialCounts[tri.materialId]++;
        }
        cout << "\nMaterial usage in mesh:" << endl;
        for (const auto& pair : materialCounts) {
            cout << "  Material " << pair.first << ": " << pair.second << " triangles";
            if (pair.first < materials.size()) {
                const Material& mat = materials[pair.first];
                cout << " (";
                if (mat.baseColorTextureIdx >= 0) cout << "baseColor ";
                if (mat.metallicRoughnessTextureIdx >= 0) cout << "metalRough ";
                if (mat.normalTextureIdx >= 0) cout << "normal ";
                if (mat.emissiveTextureIdx >= 0) cout << "emissive ";
                if (mat.occlusionTextureIdx >= 0) cout << "occlusion ";
                cout << "textures)";
            }
            cout << endl;
        }
    }

    cout << "========================\n" << endl;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        newMaterial.baseColorTextureIdx = -1;
        newMaterial.metallicRoughnessTextureIdx = -1;
        newMaterial.normalTextureIdx = -1;
        newMaterial.emissiveTextureIdx = -1;
        newMaterial.occlusionTextureIdx = -1;

        // Initialize other material properties
        newMaterial.emittance = 0.0f;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emissiveFactor = glm::vec3(0.0f);
        

        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.transparency = 0.0f;  
            newMaterial.roughness = 1.0f;     
            newMaterial.metallic = 0.0f;      

			newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
			newMaterial.type = EMITTING;
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.transparency = 0.0f;
            newMaterial.roughness = 0.0f;
            newMaterial.metallic = 0.2f;

			newMaterial.type = SPECULAR;
            }
        else if (p["TYPE"] == "Refractive")
        {
            newMaterial.transparency = 1.0f;
            newMaterial.roughness = 0.0f;
            newMaterial.metallic = 0.0f;

            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = p["IOR"];
            }
            else {
                newMaterial.indexOfRefraction = 1.5f;  // Default glass IOR
            }
            newMaterial.type = REFRACTIVE;
        }
        else if (p["TYPE"] == "PBR")
        {
            // Initialize PBR properties with default values
            newMaterial.transparency = 0.0f;  // Default: fully opaque
            newMaterial.roughness = 0.5f;     // Default: medium roughness
            newMaterial.metallic = 0.0f;      // Default: non-metallic

            // Parse transparency (0.0 = fully opaque, 1.0 = fully transparent)
            if (p.contains("TRANSPARENCY")) {
                newMaterial.transparency = p["TRANSPARENCY"];
                // Clamp to valid range
                newMaterial.transparency = glm::clamp(newMaterial.transparency, 0.0f, 1.0f);

                if (p.contains("IOR")) {
                    newMaterial.indexOfRefraction = p["IOR"];
                }
                else {
                    newMaterial.indexOfRefraction = 1.5f;  // Default glass IOR
                }

            }

            // Parse roughness (0.0 = perfectly smooth/mirror, 1.0 = completely rough/diffuse)
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
                // Clamp to valid range
                newMaterial.roughness = glm::clamp(newMaterial.roughness, 0.0f, 1.0f);
            }

            // Parse metallic (0.0 = dielectric/non-metal, 1.0 = metal)
            if (p.contains("METALLIC")) {
                newMaterial.metallic = p["METALLIC"];
                // Clamp to valid range
                newMaterial.metallic = glm::clamp(newMaterial.metallic, 0.0f, 1.0f);
            }

            if (p.contains("SUBSURFACE_ENABLED")) {
                newMaterial.subsurfaceEnabled = p["SUBSURFACE_ENABLED"];

                if (newMaterial.subsurfaceEnabled > 0) {
                    // Load subsurface color
                    if (p.contains("SUBSURFACE_COLOR")) {
                        const auto& sssColor = p["SUBSURFACE_COLOR"];
                        newMaterial.subsurfaceColor = glm::vec3(
                            sssColor[0], sssColor[1], sssColor[2]
                        );
                    }
                    else {
                        // Default to slightly tinted version of base color
                        newMaterial.subsurfaceColor = newMaterial.color * glm::vec3(0.8f);
                    }

                    // Load per-channel subsurface radius
                    if (p.contains("SUBSURFACE_RADIUS")) {
                        const auto& sssRadius = p["SUBSURFACE_RADIUS"];
                        if (sssRadius.is_array() && sssRadius.size() == 3) {
                            newMaterial.subsurfaceRadiusRGB = glm::vec3(
                                sssRadius[0], sssRadius[1], sssRadius[2]
                            );
                        }
                        else if (sssRadius.is_number()) {
                            float radius = sssRadius;
                            newMaterial.subsurfaceRadiusRGB = glm::vec3(radius);
                        }
                    }
                    else {
                        // Default radius
                        newMaterial.subsurfaceRadiusRGB = glm::vec3(0.01f);
                    }

                    // Load subsurface scale
                    if (p.contains("SUBSURFACE_SCALE")) {
                        newMaterial.subsurfaceScale = p["SUBSURFACE_SCALE"];
                    }
                    else {
                        newMaterial.subsurfaceScale = 1.0f;
                    }

                    // Load subsurface anisotropy
                    if (p.contains("SUBSURFACE_ANISOTROPY")) {
                        newMaterial.subsurfaceAnisotropy = glm::clamp(
                            (float)p["SUBSURFACE_ANISOTROPY"], -1.0f, 1.0f
                        );
                    }
                    else {
                        newMaterial.subsurfaceAnisotropy = 0.0f;
                    }

                    if (p.contains("SUBSURFACE_RADIUS_SINGLE")) {
                        float radius = p["SUBSURFACE_RADIUS_SINGLE"];
                        newMaterial.subsurfaceRadius = radius;
                        if (!p.contains("SUBSURFACE_RADIUS")) {
                            newMaterial.subsurfaceRadiusRGB = glm::vec3(radius);
                        }
                    }
                    else {
                        newMaterial.subsurfaceRadius = glm::length(
                            newMaterial.subsurfaceRadiusRGB
                        ) / 1.732f; // Average of RGB channels
                    }

                    std::cout << "  SSS Enabled: color("
                        << newMaterial.subsurfaceColor.x << ", "
                        << newMaterial.subsurfaceColor.y << ", "
                        << newMaterial.subsurfaceColor.z << "), "
                        << "radius("
                        << newMaterial.subsurfaceRadiusRGB.x << ", "
                        << newMaterial.subsurfaceRadiusRGB.y << ", "
                        << newMaterial.subsurfaceRadiusRGB.z << "), "
                        << "scale: " << newMaterial.subsurfaceScale << ", "
                        << "anisotropy: " << newMaterial.subsurfaceAnisotropy
                        << std::endl;
                }
            }
            else {
                // SSS disabled by default
                newMaterial.subsurfaceEnabled = 0;
                newMaterial.subsurfaceColor = glm::vec3(1.0f);
                newMaterial.subsurfaceRadius = 0.01f;
                newMaterial.subsurfaceRadiusRGB = glm::vec3(0.01f);
                newMaterial.subsurfaceScale = 1.0f;
                newMaterial.subsurfaceAnisotropy = 0.0f;
            }

			newMaterial.type = PBR;
        }


        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    // Load environment map if specified
    if (data.contains("EnvironmentMap"))
    {
        const auto& envMapData = data["EnvironmentMap"];
        if (envMapData.contains("FILE"))
        {
            std::string envMapFile = envMapData["FILE"];

            // Handle both absolute and relative paths
            std::string fullPath = envMapFile;

            // If relative path, try relative to the scene file location
            if (envMapFile[0] != '/' && envMapFile[1] != ':')  // Not an absolute path
            {
                size_t lastSlash = jsonName.find_last_of("/\\");
                if (lastSlash != std::string::npos)
                {
                    std::string sceneDir = jsonName.substr(0, lastSlash + 1);
                    fullPath = sceneDir + envMapFile;
                }
            }

            cout << "Loading environment map from: " << fullPath << endl;

            // Load HDR image using stb_image
            int width, height, channels;
            float* hdrData = stbi_loadf(fullPath.c_str(), &width, &height, &channels, 3);

            if (hdrData)
            {
                environmentMap.enabled = true;
                environmentMap.width = width;
                environmentMap.height = height;

                // Get intensity multiplier if specified
                if (envMapData.contains("INTENSITY"))
                {
                    environmentMap.intensity = envMapData["INTENSITY"];
                }
                else
                {
                    environmentMap.intensity = 1.0f;
                }

                // Convert to glm::vec3 and store
                environmentMap.data.resize(width * height);
                for (int i = 0; i < width * height; ++i)
                {
                    environmentMap.data[i] = glm::vec3(
                        hdrData[i * 3 + 0],
                        hdrData[i * 3 + 1],
                        hdrData[i * 3 + 2]
                    ) * environmentMap.intensity;
                }

                // Free the original HDR data
                stbi_image_free(hdrData);

                cout << "Environment map loaded: " << width << "x" << height << " pixels" << endl;
                cout << "Environment map intensity: " << environmentMap.intensity << endl;
            }
            else
            {
                cout << "Failed to load environment map: " << fullPath << endl;
                cout << "Continuing without environment map..." << endl;
            }
        }
    }

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.meshData = nullptr;
        newGeom.triangleStart = -1;  // Initialize to invalid
        newGeom.triangleCount = 0;   // Initialize to 0

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (type == "cube")
        {
            newGeom.type = CUBE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
        }
        else if (type == "gltf")
        {
            newGeom.type = GLTF_MESH;

            std::string gltfFile = p["FILE"];
            std::string fullPath = gltfFile;
            if (gltfFile[0] != '/' && gltfFile[1] != ':')  // Not an absolute path
            {
                size_t lastSlash = jsonName.find_last_of("/\\");
                if (lastSlash != std::string::npos)
                {
                    std::string sceneDir = jsonName.substr(0, lastSlash + 1);
                    fullPath = sceneDir + gltfFile;
                }
            }
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);

            // Load GLTF - it will handle its own materials
            loadGLTFModel(fullPath, newGeom, translation, rotation, scaleVec);
        }
        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
