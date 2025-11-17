/**
 * @file modelcreator.h
 * @brief Declaration of the ModelCreator class for generating and mutating FFN model structures.
 *
 * @details
 * The ModelCreator class is responsible for:
 *
 * - Representing network structures as linear integer parameter vectors
 * - Translating integer vectors into @ref CModelStructure or @ref CModelStructure_Multi objects
 * - Creating random network structures (for RMS mode)
 * - Decoding GA chromosomes (vector<long unsigned int>) into model architectures
 * - Computing model fitness (MSE, combined metrics)
 * - Interfacing with FFNWrapper_Multi for training/evaluation
 *
 * ModelCreator is the core utility used by:
 * - **Genetic Algorithm (GA)** for evolving model structures
 * - **Random Model Structure search (RMS)** for stochastic architecture exploration
 * - **Manual/Deterministic mode** through BuildModelStructure()
 *
 * It stores constraints such as:
 * - Maximum number of layers
 * - Maximum nodes per layer
 * - Maximum lag depth
 * - Input column count
 *
 * @see FFNWrapper_Multi
 * @see CModelStructure
 * @see CModelStructure_Multi
 */

#ifndef MODELCREATOR_H
#define MODELCREATOR_H

#include <vector>
#include <map>
#include <QString>
#include <gsl/gsl_rng.h>

#include "ffnwrapper.h"
#include "ffnwrapper_multi.h"
#include <BTCSet.h>

using namespace std;

/**
 * @class ModelCreator
 * @brief Generates, mutates, and decodes FFN model structures from parameter vectors.
 *
 * @details
 * ModelCreator provides a **mapping between evolutionary parameters** (integers)
 * and real neural network architectures (layers, nodes, lags, input columns).
 *
 * ### Key Responsibilities
 * - Random model structure generation
 * - GA chromosome decoding
 * - Assigning parameters to CModelStructure and CModelStructure_Multi
 * - Creating FFNWrapper models from generated structures
 * - Computing fitness using FFNWrapper_Multi::PerformanceMetrics()
 *
 * ### Parameter Encoding
 * The internal `parameters` vector is used by GA or RMS mode.
 * Each element corresponds to a structural feature (nodes, lags, multiplier, etc.)
 *
 * ### RNG
 * Uses GSL Tausworthe generator for reproducibility.
 *
 * @note
 *   This class is stateful. ModelCreator objects used in GA **must not be const**.
 *
 * @warning
 *   Do not access the internal RNG pointer `r` directly.
 */
class ModelCreator
{
public:

    /**
     * @brief Pointer to the actively constructed model structure.
     *
     * @note
     * This is usually filled inside CreateModel() or CreateRandomModelStructure().
     */
    CModelStructure_Multi *modelstruct = nullptr;

    /** @brief Default constructor. Initializes RNG and empty parameter vector. */
    ModelCreator();

    /** @brief Copy constructor. Performs deep copy of parameters and settings. */
    ModelCreator(const ModelCreator &other);

    /** @brief Assignment operator. Copies internal parameters and limits. */
    ModelCreator &operator=(const ModelCreator &other);

    /**
     * @brief Set the integer parameter vector.
     *
     * @param params A vector of integer parameters (GA chromosome).
     * @return true if successfully stored, false otherwise.
     *
     * @note
     * Does not immediately create a model. Use AssignParameters() or CreateModel().
     */
    bool SetParameters(const vector<long int> &params)
    {
        parameters = params;
        return true;
    }

    /**
     * @brief Get the number of structural parameters encoded.
     *
     * @return Number of elements in internal parameter vector.
     */
    int ParametersSize();

    /**
     * @brief Populate parameters based on an existing @ref CModelStructure.
     *
     * @param modelstructure The source model structure.
     * @return true if successful.
     */
    bool SetParameters(CModelStructure *modelstructure);

    /**
     * @brief Decode the internal parameter vector into a CModelStructure.
     *
     * @param modelstructure Output pointer to be overwritten.
     * @return true if decoding is valid.
     *
     * @note
     * This version is for single-output models.
     */
    bool CreateModel(CModelStructure *modelstructure) const;

    /**
     * @brief Create a multi-output model structure from parameters.
     *
     * @param modelstructure Output pointer to fill.
     * @return true if successful.
     */
    bool CreateModel(CModelStructure_Multi *modelstructure);

    /**
     * @brief Create a random model structure (single-output).
     *
     * @param modelstructure Output pointer.
     * @return true if random generation succeeds.
     *
     * @note
     * Uses internal RNG + structural limits.
     */
    bool CreateRandomModelStructure(CModelStructure *modelstructure);

    /**
     * @brief Create a random model structure (multi-output).
     *
     * @param modelstructure Output pointer.
     * @return true if random structure is valid.
     *
     * @note
     * This is the function used during RMS mode.
     */
    bool CreateRandomModelStructure(CModelStructure_Multi *modelstructure);

    /**
     * @brief Determine maximum possible parameter value for parameter index i.
     *
     * @param i Parameter index.
     * @return Largest allowed integer value for that gene.
     *
     * @note
     * Used by GA to determine chromosome search bounds.
     */
    unsigned long int MaxParameter(int i);

    /**
     * @brief Compute fitness (MSE or combined metric) for current model.
     *
     * @return A map<string,double> containing fitness values.
     *
     * @note
     * Keys may include: "MSE_Test", "MSE_Train", "CombinedLoss", etc.
     */
    map<string, double> Fitness();

    /**
     * @brief Assign GA chromosome to internal parameters vector.
     *
     * @param x Chromosome vector (unsigned ints).
     *
     * @details
     * After calling this, call CreateModel() to decode structure.
     */
    void AssignParameters(const vector<long unsigned int> &x);

    /**
     * @brief Create a model using internal parameter vector and stored modelstruct.
     *
     * @return true if successful.
     *
     * @note
     * ModelCreator must have modelstruct != nullptr.
     */
    bool CreateModel();

    /**
     * @brief The FFN wrapper used for training/evaluation of generated models.
     *
     * @note
     * Required for GA fitness calculation.
     */
    FFNWrapper_Multi FFN;

    /**
     * @brief Append the current model structure to a results file.
     *
     * @return true on success.
     */
    bool AppendModelStructureToFile();

    /**
     * @brief Number of input columns available (excluding outputs).
     *
     * @note
     * Set in main.cpp:
     * `total_number_of_columns = total_data_cols - number_of_outputs`
     */
    int total_number_of_columns = 0;

    /** @brief Whether initialization has been completed. */
    bool initiated = false;

    /** @brief Maximum possible time lag (in discrete steps). */
    int maximum_superficial_lag = 0;

    /** @brief Frequency of lag sampling (e.g., every 3 steps). */
    int lag_frequency = 0;

    /** @brief Maximum allowable nodes in a hidden layer. */
    int max_number_of_nodes_in_layers = 10;

    /** @brief Maximum allowable number of hidden layers. */
    int max_number_of_layers = 4;

    /** @brief Maximum lag multiplier for random lag generation. */
    int max_lag_multiplier = 6;

    /**
     * @brief Clear all fields of a single-output model structure.
     *
     * @param modelstructure Pointer to clear.
     */
    void clear(CModelStructure *modelstructure);

    /**
     * @brief Clear all fields of a multi-output model structure.
     *
     * @param modelstructure Pointer to clear.
     */
    void clear(CModelStructure_Multi *modelstructure);

    /**
     * @brief Equality operator comparing decoded model structures.
     *
     * @param m2 Another ModelCreator instance.
     * @return true if both generate identical CModelStructure instances.
     *
     * @note
     * Does not compare raw parameters; compares structural meaning.
     */
    bool operator==(const ModelCreator &m2)
    {
        CModelStructure modstruct1;
        CModelStructure modstruct2;
        CreateModel(&modstruct1);
        m2.CreateModel(&modstruct2);
        return (modstruct1 == modstruct2);
    }

    /**
     * @brief Inequality operator (logical negation of operator==).
     */
    bool operator!=(const ModelCreator &m2)
    {
        return (!operator==(m2));
    }

private:

    /**
     * @brief Vector of encoded architecture parameters.
     *
     * @details
     * This vector represents:
     * - Number of layers
     * - Nodes in each layer
     * - Lag multipliers
     * - Input selections
     *
     * It is decoded by CreateModel() or randomly generated.
     */
    vector<long int> parameters;

    /**
     * @brief GSL random number generator (Tausworthe).
     *
     * @note Allocated on construction, freed on program exit.
     */
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
};

/**
 * @brief Convert an integer to an arbitrary-base representation.
 *
 * @param number The integer to convert.
 * @param base The base into which to convert (e.g., base 10, 16, 3).
 * @return A vector of digits in the given base (least-significant digit first).
 *
 * @details
 * Useful for genetic algorithms and grid-search encodings.
 */
std::vector<int> convertToBase(unsigned long int number, int base);

#endif // MODELCREATOR_H
