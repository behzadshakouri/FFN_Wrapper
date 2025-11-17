/**
 * @file trainer.h
 * @brief Declares all training-mode functions used by the FFN Wrapper.
 *
 * @details
 * This header exposes the three primary training procedures:
 *
 * 1. @ref RunGA()
 *    - Performs Genetic Algorithm optimization of the model structure
 *    - Evaluates candidates using MSE-Test or combined metrics
 *    - Saves optimized structure and dataset predictions
 *
 * 2. @ref RunRandom()
 *    - Generates random model structures using ModelCreator
 *    - Trains and evaluates each one
 *    - Writes results to RMS_Output.txt
 *
 * 3. @ref RunSingle()
 *    - Runs a single user-defined model structure (manual or GA-based)
 *    - Used when GA and Random modes are disabled
 *
 * ### Design Goals
 * - Keep main.cpp clean
 * - Separate model setup (modelbuilder) from training logic
 * - Allow each training mode to share the same CModelStructure_Multi
 *
 * ### Integration
 * The training functions require:
 * - @ref Config (non-const reference)
 * - @ref CModelStructure_Multi (input+output paths, structure, etc.)
 *
 * @note
 * These functions do not configure the network structure;
 * that is done in BuildModelStructure().
 *
 * @see trainer.cpp
 * @see modelbuilder.cpp
 * @see BuildModelStructure()
 * @see BuildAddresses()
 */

#pragma once

#include "config.h"
#include "ffnwrapper_multi.h"

/**
 * @brief Run Genetic Algorithm optimization of model structure.
 *
 * @details
 * Executes a full GA cycle using:
 * - cfg.GA_Nsim for number of generations
 * - cfg.MSE_Test as optimization criterion
 * - cfg.modelCreator as the generator/decoder of structures
 *
 * After GA finishes:
 * - Saves Train/Test predictions
 * - Writes GA results to *GA_results.txt*
 *
 * @param ms   Model structure to initialize/override and evaluate.
 * @param cfg  Configuration (non-const because ModelCreator may mutate).
 *
 * @note Requires cfg.GA_switch = true in main.cpp.
 */
void RunGA(CModelStructure_Multi& ms, Config& cfg);

/**
 * @brief Run random model structure search.
 *
 * @details
 * Creates @c cfg.Random_Nsim random model structures using:
 * - cfg.modelCreator.CreateRandomModelStructure()
 * - CModelStructure_Multi::ValidLags() to reject invalid models
 *
 * For each valid structure:
 * - Initializes FFN
 * - Performs Train or Train_kfold
 * - Runs Test + PerformanceMetrics
 * - Saves Train/Test predictions
 * - Writes structure info to "RMS_Output.txt"
 *
 * @param ms   Model structure (initial content overwritten for each random run).
 * @param cfg  Configuration (modelCreator is mutated by random generation).
 *
 * @warning
 * This mode easily produces hundreds/thousands of runs.
 * Ensure cfg.Random_Nsim is set responsibly.
 */
void RunRandom(CModelStructure_Multi& ms, Config& cfg);

/**
 * @brief Train and evaluate a single model structure.
 *
 * @details
 * This is the simplest training mode.
 * Steps:
 * - Initialize FFNWrapper_Multi with @c ms
 * - Train via Train() or Train_kfold()
 * - Run Test()
 * - Compute PerformanceMetrics()
 * - Save training and test predictions
 * - Creates plots via F.Plotter()
 *
 * @param ms   The model structure built in BuildModelStructure().
 * @param cfg  Configuration containing K-fold and paths.
 *
 * @note
 * This is used when:
 * - GA_switch == false
 * - randommodelstructure == false
 */
void RunSingle(CModelStructure_Multi& ms, Config& cfg);
