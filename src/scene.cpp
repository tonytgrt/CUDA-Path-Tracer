#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

// Add STB Image implementation for HDR loading
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
			newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
			newMaterial.type = EMITTING;
        }
        else if (p["TYPE"] == "Specular")
        {
			newMaterial.type = SPECULAR;
            }
        else if (p["TYPE"] == "Refractive")
        {
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
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
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
