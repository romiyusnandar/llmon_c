#include "../include/tensor.h"
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

Tensor::Tensor(int r, int c) : rows(r), cols(c) {
    data.resize(r * c, 0.0f);
    grad.resize(r * c, 0.0f);
    _backward = [](){};
}

TensorPtr Tensor::create(int r, int c) {
    return std::make_shared<Tensor>(r, c);
}

void Tensor::random_init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (rows + cols));
    std::uniform_real_distribution<> dis(-limit, limit);
    for (auto& val : data) val = dis(gen);
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

float& Tensor::at(int i, int j) { return data[i * cols + j]; }
float& Tensor::grad_at(int i, int j) { return grad[i * cols + j]; }

void Tensor::backward() {
    std::vector<Tensor*> topo;
    std::set<Tensor*> visited;

    std::function<void(Tensor*)> build_topo = [&](Tensor* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto child : v->prev) {
                build_topo(child.get());
            }
            topo.push_back(v);
        }
    };

    build_topo(this);

    std::fill(grad.begin(), grad.end(), 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

void Tensor::print() const {
    std::cout << "Tensor (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(4) << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void Tensor::print_grad() const {
    std::cout << "Gradient (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(4) << grad[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

float Tensor::mean() const {
    if (data.empty()) return 0.0f;
    float sum = 0.0f;
    for (const auto& val : data) sum += val;
    return sum / data.size();
}

float Tensor::std_dev() const {
    if (data.empty()) return 0.0f;
    float m = mean();
    float variance = 0.0f;
    for (const auto& val : data) {
        float diff = val - m;
        variance += diff * diff;
    }
    return std::sqrt(variance / data.size());
}