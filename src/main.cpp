#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static std::string startTimeString;

static std::string currentSceneFile = "";
static char filePathBuffer[256] = "";
static bool needsReload = false;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

static int guiWidth = 400;  // Width of the ImGui panel
static int windowWidth;      // Total window width
static int windowHeight;     // Total window height
static bool firstFrame = true;
static float zoomSpeed = 0.1f;


float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    // Create window with extra width for GUI panel
    windowWidth = width + guiWidth;
    windowHeight = height;

    window = glfwCreateWindow(windowWidth, windowHeight, "CUDA Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));

    //Set up ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Use a clean, modern style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.GrabRounding = 4.0f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}

void reloadScene(const std::string& sceneFilePath) {
    // Save the current scene file path
    currentSceneFile = sceneFilePath;

    // Clean up the old scene
    if (scene != nullptr) {
        pathtraceFree();
        delete scene;
        scene = nullptr;
    }

    // Load the new scene
    scene = new Scene(sceneFilePath.c_str());

    // Reset camera and render state
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // Recalculate camera parameters
    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Reinitialize path tracer
    pathtraceInit(scene);

    // Force camera update
    camchanged = true;

    printf("Scene reloaded: %s\n", sceneFilePath.c_str());
}

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Set up the ImGui window to dock on the right side
    if (firstFrame) {
        ImGui::SetNextWindowPos(ImVec2(width, 0));
        ImGui::SetNextWindowSize(ImVec2(guiWidth, windowHeight));
        firstFrame = false;
    }

    // Create a window that fills the right side
    ImGui::SetNextWindowPos(ImVec2(width, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(guiWidth, windowHeight), ImGuiCond_Always);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("Path Tracer Control Panel", nullptr, window_flags);

    // Scene Controls Section
    ImGui::Separator();
    ImGui::Text("Scene Controls");
    ImGui::Separator();

    // Display current scene file (truncate if too long)
    std::string displayPath = currentSceneFile;
    if (displayPath.length() > 40) {
        displayPath = "..." + displayPath.substr(displayPath.length() - 37);
    }
    ImGui::Text("Current: %s", displayPath.c_str());

    // Reload button
    if (ImGui::Button("Reload Scene", ImVec2(-1, 30))) {
        if (!currentSceneFile.empty()) {
            reloadScene(currentSceneFile);
        }
    }

    ImGui::Spacing();

    // File picker section
    ImGui::Text("Load New Scene:");
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##filepath", filePathBuffer, sizeof(filePathBuffer));
    ImGui::PopItemWidth();

    if (ImGui::Button("Load Scene", ImVec2(-1, 30))) {
        if (strlen(filePathBuffer) > 0) {
            std::string newScenePath(filePathBuffer);

            // Check if file exists
            std::ifstream testFile(newScenePath);
            if (testFile.good()) {
                testFile.close();
                reloadScene(newScenePath);
            }
            else {
                ImGui::OpenPopup("Error");
            }
        }
    }

    // Error popup
    if (ImGui::BeginPopup("Error")) {
        ImGui::Text("Could not open scene file!");
        ImGui::Text("%s", filePathBuffer);
        if (ImGui::Button("OK", ImVec2(80, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Rendering Stats Section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Rendering Statistics");
    ImGui::Separator();

    ImGui::Text("Iteration: %d / %d", iteration, renderState->iterations);
    ImGui::Text("Trace Depth: %d", imguiData->TracedDepth);
    ImGui::Text("Resolution: %d x %d", width, height);

    // Progress bar for iterations
    float progress = (float)iteration / (float)renderState->iterations;
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
    ImGui::Text("Progress: %.1f%%", progress * 100.0f);

    ImGui::Spacing();
    ImGui::Text("Performance:");
    ImGui::BulletText("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
    ImGui::BulletText("%.1f FPS", ImGui::GetIO().Framerate);

    // Camera Controls Section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Camera Controls");
    ImGui::Separator();

    if (ImGui::Button("Reset Camera", ImVec2(-1, 25))) {
        camchanged = true;
        Camera& cam = renderState->camera;
        cam.lookAt = ogLookAt;
        zoom = glm::length(cam.position - ogLookAt);  // Reset zoom too
        iteration = 0;
    }

    ImGui::Text("Position:");
    ImGui::BulletText("X: %.2f", cameraPosition.x);
    ImGui::BulletText("Y: %.2f", cameraPosition.y);
    ImGui::BulletText("Z: %.2f", cameraPosition.z);

    ImGui::Spacing();
    ImGui::Text("Zoom: %.2f", zoom);
    ImGui::SliderFloat("Zoom Speed", &zoomSpeed, 0.01f, 0.5f, "%.2f");

    // Quick zoom buttons
    ImGui::Text("Quick Zoom:");
    ImGui::SameLine();
    if (ImGui::Button("-", ImVec2(30, 0))) {
        zoom *= 1.2f;
        zoom = std::fmin(zoom, 100.0f);
        camchanged = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(60, 0))) {
        zoom = glm::length(scene->state.camera.position - scene->state.camera.lookAt);
        camchanged = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+", ImVec2(30, 0))) {
        zoom *= 0.8f;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }

    // Quick Actions Section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Quick Actions");
    ImGui::Separator();

    if (ImGui::Button("Save Image", ImVec2(-1, 30))) {
        saveImage();
        ImGui::OpenPopup("Image Saved");
    }

    // Image saved popup
    if (ImGui::BeginPopupModal("Image Saved", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Image saved successfully!");
        ImGui::Separator();
        ImGui::Text("File: %s", renderState->imageName.c_str());
        ImGui::Spacing();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Instructions (collapsible)
    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Keyboard & Mouse Controls")) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Keyboard:");
        ImGui::BulletText("[S] Save Image");
        ImGui::BulletText("[Space] Reset Camera");
        ImGui::BulletText("[Esc] Save & Exit");
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Mouse:");
        ImGui::BulletText("[LMB Drag] Rotate Camera");
        ImGui::BulletText("[Scroll] Zoom In/Out");
        ImGui::BulletText("[RMB Drag] Alternative Zoom");
        ImGui::BulletText("[MMB Drag] Pan Camera");
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CUDA Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());

        // Set viewport to only render in the left portion of the window
        glViewport(0, 0, width, height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        // Clear the entire window first
        glViewport(0, 0, windowWidth, windowHeight);
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render the path traced image only in the left viewport
        glViewport(0, 0, width, height);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        // Reset viewport for ImGui
        glViewport(0, 0, windowWidth, windowHeight);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];
    currentSceneFile = std::string(sceneFile); 
    strncpy(filePathBuffer, sceneFile, sizeof(filePathBuffer) - 1);

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}



void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    // Get current mouse position to check if it's over the render area
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Only process scroll if mouse is in the rendering area (left side)
    // and not over ImGui
    if (xpos > width || MouseOverImGuiWindow()) {
        return;
    }

    // Adjust zoom based on scroll direction
    // yoffset is positive when scrolling up (zoom in), negative when scrolling down (zoom out)
    float zoomDelta = -yoffset * zoomSpeed;
    zoom *= (1.0f + zoomDelta);

    // Clamp zoom to reasonable values
    zoom = std::fmax(0.1f, std::fmin(zoom, 100.0f));

    // Mark camera as changed to trigger re-render
    camchanged = true;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Check if mouse is over the ImGui window or outside render area
    if (MouseOverImGuiWindow() || xpos > width)
    {
        leftMousePressed = false;
        rightMousePressed = false;
        middleMousePressed = false;
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}


void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    // Only process mouse input if it's in the rendering area (left side)
    // and not over ImGui
    if (xpos > width || MouseOverImGuiWindow()) {
        lastX = xpos;
        lastY = ypos;
        return;
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        // Alternative zoom method with right mouse button (finer control)
        float zoomDelta = (ypos - lastY) / height;
        zoom *= (1.0f + zoomDelta);
        zoom = std::fmax(0.1f, std::fmin(zoom, 100.0f));
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}


