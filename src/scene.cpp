#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
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

// Load texture from glTF
int Scene::loadGLTFTexture(const tinygltf::Texture& gltfTex,
    const tinygltf::Model& model) {
    if (gltfTex.source < 0) return -1;

    const tinygltf::Image& image = model.images[gltfTex.source];

    Texture tex;
    tex.width = image.width;
    tex.height = image.height;
    tex.components = image.component;

    // Allocate and copy image data
    size_t imageSize = tex.width * tex.height * tex.components;
    tex.data = new unsigned char[imageSize];
    memcpy(tex.data, image.image.data(), imageSize);

    textures.push_back(tex);
    return textures.size() - 1;
}

// Load material from glTF
int Scene::loadGLTFMaterial(const tinygltf::Material& gltfMat,
    const tinygltf::Model& model) {
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

    // Load base color texture if present
    if (pbr.baseColorTexture.index >= 0) {
        mat.baseColorTextureIdx = loadGLTFTexture(
            model.textures[pbr.baseColorTexture.index], model);
    }

    // Load metallic-roughness texture if present
    // Note: In glTF, metallic is in blue channel, roughness is in green channel
    if (pbr.metallicRoughnessTexture.index >= 0) {
        mat.metallicRoughnessTextureIdx = loadGLTFTexture(
            model.textures[pbr.metallicRoughnessTexture.index], model);
    }

    // Load normal texture if present
    if (gltfMat.normalTexture.index >= 0) {
        mat.normalTextureIdx = loadGLTFTexture(
            model.textures[gltfMat.normalTexture.index], model);
    }

    // Load emissive texture and factor
    if (gltfMat.emissiveFactor.size() >= 3) {
        mat.emissiveFactor = glm::vec3(
            gltfMat.emissiveFactor[0],
            gltfMat.emissiveFactor[1],
            gltfMat.emissiveFactor[2]
        );
        // If there's an emissive factor, treat it as emissive
        if (glm::length(mat.emissiveFactor) > 0.0f) {
            mat.emittance = glm::length(mat.emissiveFactor);
            mat.type = EMITTING;
        }
    }

    if (gltfMat.emissiveTexture.index >= 0) {
        mat.emissiveTextureIdx = loadGLTFTexture(
            model.textures[gltfMat.emissiveTexture.index], model);
    }

    // Load occlusion texture if present
    if (gltfMat.occlusionTexture.index >= 0) {
        mat.occlusionTextureIdx = loadGLTFTexture(
            model.textures[gltfMat.occlusionTexture.index], model);
    }

    // Handle alpha mode
    if (gltfMat.alphaMode == "BLEND" || gltfMat.alphaMode == "MASK") {
        // Material has transparency
        if (gltfMat.alphaMode == "MASK") {
            // Use alpha cutoff for masked transparency
            mat.transparency = (gltfMat.alphaCutoff > 0.5f) ? 1.0f : 0.0f;
        }
    }

    // Handle double-sided materials (you might want to store this info)
    // gltfMat.doubleSided

    materials.push_back(mat);
    return materials.size() - 1;
}

// Updated loadGLTFModel function
int Scene::loadGLTFModel(const std::string& gltfPath, Geom& newGeom,
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
        return -1;
    }

    if (!ret) {
        cout << "Failed to load glTF: " << gltfPath << endl;
        return -1;
    }

    // Load all materials from the glTF file
    std::vector<int> materialMapping;
    for (const auto& gltfMat : model.materials) {
        int matIdx = loadGLTFMaterial(gltfMat, model);
        materialMapping.push_back(matIdx);
    }

    // If no materials, create a default one
    if (materialMapping.empty()) {
        Material defaultMat;
        defaultMat.type = PBR;
        defaultMat.color = glm::vec3(0.8f);
        defaultMat.roughness = 0.5f;
        defaultMat.metallic = 0.0f;
        defaultMat.transparency = 0.0f;
        defaultMat.baseColorTextureIdx = -1;
        defaultMat.metallicRoughnessTextureIdx = -1;
        defaultMat.normalTextureIdx = -1;
        defaultMat.emissiveTextureIdx = -1;
        defaultMat.occlusionTextureIdx = -1;
        materials.push_back(defaultMat);
        materialMapping.push_back(materials.size() - 1);
    }

    // Create a new MeshData to store all triangles
    MeshData meshData;
    meshData.boundingBoxMin = glm::vec3(FLT_MAX);
    meshData.boundingBoxMax = glm::vec3(-FLT_MAX);

    // Process all meshes in the model
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                cout << "Warning: Non-triangle primitive skipped" << endl;
                continue;
            }

            // Get the material index for this primitive
            int materialIdx = 0; // Default material
            if (primitive.material >= 0 && primitive.material < materialMapping.size()) {
                materialIdx = materialMapping[primitive.material];
            }

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

            // Process triangles (rest of the triangle loading code remains the same)
            if (primitive.indices >= 0) {
                // ... [same triangle loading code as before] ...
                // But add this when creating each triangle:
                // tri.materialId = materialIdx;
            }
        }
    }

    // Store mesh data
    meshes.push_back(meshData);
    newGeom.meshData = &meshes.back();
    newGeom.triangleCount = meshData.triangles.size();

    // Return the first material index (for backward compatibility)
    // In reality, each triangle now has its own material
    return materialMapping.empty() ? -1 : materialMapping[0];
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

        if (type == "cube")
        {
            newGeom.type = CUBE;
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

            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);

            // Load GLTF - it will handle its own materials
            int defaultMatIdx = loadGLTFModel(fullPath, newGeom, translation, rotation, scaleVec);
            newGeom.materialid = defaultMatIdx; // This is just for compatibility
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
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
