#pragma once

#include "sceneStructs.h"
#include <vector>
#include "tiny_gltf.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTFModel(const std::string& gltfPath, Geom& newGeom,
        const glm::vec3& translation,
        const glm::vec3& rotation,
        const glm::vec3& scale);
    int loadGLTFMaterial(const tinygltf::Material& gltfMat,
        const tinygltf::Model& model);
    int loadGLTFTexture(const tinygltf::Texture& gltfTex,
        const tinygltf::Model& model);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<MeshData> meshes;
    std::vector<Texture> textures;
    RenderState state;
    HostEnvironmentMap environmentMap;
};
