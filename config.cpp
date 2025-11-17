/**
 * @file config.cpp
 * @brief Implements helper functions related to the Config structure.
 *
 * @details
 * Currently this file contains a single helper function:
 * - BuildAddresses()
 *
 * This function generates all required file paths inside @c CModelStructure_Multi
 * based on the parameters stored in @c Config.
 *
 * Responsibilities:
 * - Constructing training/test paths for ASM and non-ASM cases
 * - Generating output directories for result storage
 * - Handling multiple realizations (if Realization > 1)
 *
 * This design keeps address-building logic separate from model logic,
 * improving maintainability and reducing clutter inside main.cpp.
 *
 * @see Config
 * @see CModelStructure_Multi
 */

#include "config.h"

/**
 * @brief Build full input/output directory and file paths for a model run.
 *
 * @details
 * This function writes into the following @c CModelStructure_Multi fields:
 * - @c trainaddress
 * - @c testaddress
 * - @c outputpath
 * - @c trainobservedaddress
 * - @c trainpredictedaddress
 * - @c testobservedaddress
 * - @c testpredictedaddress
 *
 * The logic depends on:
 * - Whether ASM model is active (cfg.ASM)
 * - Constituent data name (cfg.data_name)
 * - Full paths from main.cpp (cfg.path, cfg.path_ASM, cfg.datapath, cfg.datapath_ASM)
 * - Number of realizations
 *
 * @note
 * If @c cfg.Realization > 1, each realization gets its own
 * output CSV for train/test predictions.
 *
 * @warning
 * This function does NOT create directories on disk.
 * Ensure that the Results/ folder exists before running.
 *
 * @param ms  Reference to a @c CModelStructure_Multi instance to populate.
 * @param cfg Configuration parameters (read-only).
 */
void BuildAddresses(CModelStructure_Multi& ms, const Config& cfg)
{
    for (int r = 0; r < cfg.Realization; r++)
    {
        // === ASM MODEL PATHS ===================================================
        if (cfg.ASM)
        {
            // Train/Test input file addresses
            ms.trainaddress.push_back(
                cfg.datapath_ASM + "observedoutput_train_" + cfg.data_name + ".txt"
            );

            ms.testaddress.push_back(
                cfg.datapath_ASM + "observedoutput_test_" + cfg.data_name + ".txt"
            );

            // Output folder
            ms.outputpath = cfg.path_ASM + "Results/";

            // Output files (Observed vs Predicted)
            ms.trainobservedaddress.push_back(
                ms.outputpath + "TrainOutputDataTS_" + std::to_string(r) + ".csv"
            );

            ms.trainpredictedaddress.push_back(
                ms.outputpath + "TrainDataPredictionTS_" + std::to_string(r) + ".csv"
            );

            ms.testobservedaddress.push_back(
                ms.outputpath + "TestOutputDataTS_" + std::to_string(r) + ".csv"
            );

            ms.testpredictedaddress.push_back(
                ms.outputpath + "TestDataPredictionTS_" + std::to_string(r) + ".csv"
            );
        }

        // === SIMPLE / SETTLING MODEL PATHS ====================================
        else
        {
            ms.trainaddress.push_back(
                cfg.datapath + "observedoutput_train_" + cfg.data_name + ".txt"
            );

            ms.testaddress.push_back(
                cfg.datapath + "observedoutput_test_" + cfg.data_name + ".txt"
            );

            ms.outputpath = cfg.path + "Results/";

            ms.trainobservedaddress.push_back(
                ms.outputpath + "TrainOutputDataTS_" + cfg.data_name + ".csv"
            );

            ms.trainpredictedaddress.push_back(
                ms.outputpath + "TrainDataPredictionTS_" + cfg.data_name + ".csv"
            );

            ms.testobservedaddress.push_back(
                ms.outputpath + "TestOutputDataTS_" + cfg.data_name + ".csv"
            );

            ms.testpredictedaddress.push_back(
                ms.outputpath + "TestDataPredictionTS_" + cfg.data_name + ".csv"
            );
        }
    }
}
