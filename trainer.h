#pragma once

#include "config.h"
#include "ffnwrapper_multi.h"

void RunGA(CModelStructure_Multi& ms, Config& cfg);
void RunRandom(CModelStructure_Multi& ms, Config& cfg);
void RunSingle(CModelStructure_Multi& ms, Config& cfg);
