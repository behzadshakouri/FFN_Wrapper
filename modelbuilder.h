/**
 * @file modelbuilder.h
 * @brief Declares the function responsible for generating the model structure
 *        (layers, nodes, lags, input/output columns) for ASM and Settling models.
 *
 * @details
 * The model-building logic is one of the most complex parts of the FFN Wrapper.
 * It determines:
 * - Which inputs are used
 * - How many hidden layers/nodes to include
 * - Lag structure for each input column
 * - Output column locations
 *
 * The CModelStructure_Multi object is fully populated based on:
 * - cfg.ASM (ASM vs. simple model)
 * - cfg.data_name (NO, NH, TKN, sCOD, VSS, ND)
 * - cfg.total_data_cols and cfg.number_of_outputs
 * - cfg.optimized_structure and GA-related flags
 *
 * Doxygen documentation is expanded in modelbuilder.cpp where the actual
 * implementation is located.
 *
 * @see BuildModelStructure()
 * @see CModelStructure_Multi
 * @see Config
 */

#pragma once

#include "config.h"
#include "ffnwrapper_multi.h"

/**
 * @brief Builds the complete model structure (layers, nodes, lags, inputs/outputs).
 *
 * @details
 * This function sets all fields inside @c CModelStructure_Multi, including:
 * - n_layers
 * - n_nodes (vector)
 * - inputcolumns
 * - outputcolumns
 * - lags (vector< vector<int> >)
 * - dt, seed, realization, flags, etc.
 *
 * Behavior differs depending on:
 * - @c cfg.ASM (ASM vs Settling model)
 * - @c cfg.data_name (decides which constituent-specific rules to apply)
 *
 * ### ASM Model:
 * Supports constituents:
 * - NO
 * - NH
 * - sCOD
 * - TKN
 * - VSS
 * - ND
 *
 * Each constituent has its own optimized architecture and lag structure
 * derived from prior GA optimization research.
 *
 * ### Non-ASM Model:
 * A simple structure with 1 hidden layer and manually defined lags.
 *
 * @param ms  Model structure object to populate.
 * @param cfg Configuration struct containing all runtime settings.
 *
 * @note This function does not interact with file paths or training/test data.
 * @warning Must be called before BuildAddresses() and before training.
 */
void BuildModelStructure(CModelStructure_Multi& ms, const Config& cfg);
