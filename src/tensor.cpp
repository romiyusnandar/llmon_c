#include "../include/tensor.h"
#include <random>
#include <cmath>
#include <iomanip>

Tensor::Tensor(int r, int c) : rows(r), cols(c) {
    data.resize(r * c, 0.0f);
    grad.resize(r * c, 0.0f);
}

void Tensor::random_init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Xavier Initialization for stabilization
    float limit = std::sqrt(6.0f / (rows + cols));
    std::uniform_real_distribution<> dis(-limit, limit);

    for (auto& val : data) {
        val = dis(gen);
    }
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

float& Tensor::at(int i, int j) {
    return data[i * cols + j];
}

void Tensor::print() const {
    std::cout << "Tensor (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // std::fixed dan std::setprecision for good output
            std::cout << std::fixed << std::setprecision(4) << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "-----------------\n";
}