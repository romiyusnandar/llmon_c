/**
 * Copyright: @romiyusnandar
 * When we using python and type 'x = torch.randn(10, 10)'
 * There's a lot going on in memory, in C++ we have to manage it manually
 *
 * We need a container to save:
 *      1. Data, weights value or activation
 *      2. Grand, gradient value for learning process
 *      3. Shape, matrix dimensions (rowa, cloumns)
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

/**
 * Simple structure for 2D Tensor (Matrix)
 */
struct Tensor {
    int rows;
    int cols;
    std::vector<float> data;
    std::vector<float> grad;

    Tensor(int r, int c) : rows(r), cols(c) {
        data.resize(r * c, 0.0f);
        grad.resize(r * c, 0.0f);
    }
/**
 * Initialize with random numbers (simple Xavier/Glorot init)
 */
    void random_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = sqrt(6.0f / (rows + cols));
        std::uniform_real_distribution<> dis(-limit, limit);

        for (int i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
    }

/**
 * Helper for accessing 2D data using 1D index
 * Since vectors are 1D, we access row i, column j with the formula: (i * cols + j)
 */
    float& at(int i, int j) {
        return data[i * cols + j];
    }

    void print() {
        std::cout << "Tensor (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << at(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "----------------\n";
    }
};

#endif