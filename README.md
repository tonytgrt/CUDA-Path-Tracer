CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yiding Tian
  *  [LinkedIn](https://linkedin.com/in/ytian1109), [Github](https://github.com/tonytgrt)
* Tested on: Windows 11 24H2, i9-13900H @ 4.1GHz, 32GB RAM, MSI Shadow RTX 5080 16GB Driver 581.15, Personal Laptop with External Desktop GPU via NVMe connector (PCIe 4.0 x4 Protocol)

## Project Overview
A CUDA-based path tracer capable of rendering globally-illuminated images for various custom scenes. 

### Feature Highlights

* Diffuse, Specular, Refractive, and PBR (Physically Based Rendering) shaders
* MIS (Multiple Importance Sampling) on diffuse and PBR materials
* Subsurface Scattering for PBR materials
* Custom environment maps and GLTF models loading with materials/textures/metallic etc.
* BVH (Bounding Volume Hierachy) data structure that enables rendering complex GLTF models with millions of polygons at a reasonable speed
* Material sorting, stream compaction, and Russian Roulette ray termination to boost performance
* Stochastic sampled anti aliasing to produce sharper renders
* Nvidia OptiX Denoiser integration, configurable in real time to enable quick preview and enhance the end result
* Enhanced ImGUI user interface with detailed real-time statistics monitoring and scenes/camera/denoiser controls

### Thumbnail Render (More to come!)

![](/img/thumb-0.png)

## Build and Run Instructions (Windows)

### Build Instructions

1. Make sure to have [CUDA](https://developer.nvidia.com/cuda-downloads), [CMake 4.x](https://cmake.org/download/), and [Visual Studio 2022](https://visualstudio.microsoft.com/) installed on your PC with a modern Nvidia GPU (20-Series or later)
2. Clone the repo. Open a terminal in the repo's root directory. Run the following commands:

```
mkdir build
cd build
cmake ..
```

3. This should create a `cis565_path_tracer.sln` file inside the `build` folder. Double click to open it in Visual Studio 2022.
4. In Visual Studio 2022's top menu bar, change the build mode from `Debug` to `Release`. This impacts the rendering performance a lot! Leaving in `Debug` mode would result in extremely slow rendering speed.
5. On the `Solution Explorer` menu to the left of the Visual Studio, right click on the `cis565_path_tracer` project and select `properties`. Then in the pop-up window, find `Configuration Properties` - `Debugging` - `Command Arguments`. Enter the starting `.json` scene configuration's relative file path here. An example would be `../scenes/chess.json`. You can change the `.json` scene file input here to configure startup scene to be rendered.
6. Click the Build icon on top of the Visual Studio. You will see the rendering program opened soon.

### Run Guide

* Drag Left mouse to rotate. Right mouse to pan. Scroll to zoom. Adjust the `Zoom Speed` bar in ImGUI window to change zoom speed.
* Enter a new file path in the ImGUI window to load a new scene without restarting the program.
* Click `Save Image` button to save current render image. The image will be saved under `build` directory in `.png` format. Upon finishing all iterations the program will automatically exit and save the image as well.
* Under `OptiX Denoiser` panel, the denoiser can be configured in real time. Change the `Blend Factor` bar to see how the denoised render compare to the original.
* Refer to current `.json` scene configuration files under `scenes` to see how to create your own scene file. The environmen maps should be in `.hdr` format under `envmaps` folder. The GLTF models should be in `.glb` format under `GLTF` folder.

## Core Features Completed

## Extended Features Implemented

## Third-party Libraries Used

## Performance Analysis

## WIP Renders

### Subsurface Scattering
![](/img/wip-13.png)

### OptiX Denoiser
See how OptiX Denoiser gives a high quality result with only 382 iterations.
![](/img/wip-12.png)

### High Poly Chessboard
![](/img/wip-11.png)

A 1.49 million triangles gltf model. With BVH this renders with around 80ms frametime.
![](/img/wip-10.png)

Previosuly without BVH, it renders with 18000ms frametime. This render has only 1771 iterations but took 9 hours.
![](/img/wip-9.png)

### Stanford Dragon
![](/img/wip-8.png)

### GLTF Mesh Model with textures
![](/img/wip-7.png)

### GLTF Mesh Model without textures
![](/img/wip-6.png)

### PBR Materials
![](/img/wip-5.png)

### Cornell Box with MIS and Environment Map
![](/img/wip-4.png)

### Environment Map
![](/img/wip-3.png)

### Cornell Box of Diffuse, Specular, and Refractive objects
![](/img/wip-2.png)

### Specular objects
![](/img/wip-1.png)

## Bloopers

### The Evil Dragon
A bug with subsurface scattering and meshes
![](/img/blooper-2.png)

### MIS Fireflies
![](/img/blooper-1.png)