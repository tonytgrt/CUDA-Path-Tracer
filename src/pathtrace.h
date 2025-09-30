#pragma once

#include "scene.h"
#include "utilities.h"

#define MATERIAL_SORTING 1  // Toggle material sorting on/off

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
