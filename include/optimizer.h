#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <cmath>

class SGD {
public:
    std::vector<TensorPtr> parameters;
    float learning_rate;

    SGD(std::vector<TensorPtr> params, float lr)
        : parameters(params), learning_rate(lr) {}

    void step() {
        for (auto& p : parameters) {
            for (size_t i = 0; i < p->data.size(); i++) {
                p->data[i] -= learning_rate * p->grad[i];
            }
        }
    }

    void zero_grad() {
        for (auto& p : parameters) {
            p->zero_grad();
        }
    }
};

/**
 * Adam Optimizer - More efficient than SGD
 * Adaptive learning rate with momentum
 */
class Adam {
public:
    std::vector<TensorPtr> parameters;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t; // timestep

    std::vector<std::vector<float>> m; // First moment (momentum)
    std::vector<std::vector<float>> v; // Second moment (RMSprop)

    Adam(std::vector<TensorPtr> params, float lr = 0.001f,
         float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : parameters(params), learning_rate(lr), beta1(b1), beta2(b2),
          epsilon(eps), t(0) {
        // Initialize moment vectors
        for (auto& p : parameters) {
            m.push_back(std::vector<float>(p->data.size(), 0.0f));
            v.push_back(std::vector<float>(p->data.size(), 0.0f));
        }
    }

    void step() {
        t++;
        float lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));

        for (size_t p_idx = 0; p_idx < parameters.size(); p_idx++) {
            auto& p = parameters[p_idx];
            for (size_t i = 0; i < p->data.size(); i++) {
                float grad = p->grad[i];

                // Update biased first moment estimate
                m[p_idx][i] = beta1 * m[p_idx][i] + (1.0f - beta1) * grad;

                // Update biased second moment estimate
                v[p_idx][i] = beta2 * v[p_idx][i] + (1.0f - beta2) * grad * grad;

                // Update parameters
                p->data[i] -= lr_t * m[p_idx][i] / (std::sqrt(v[p_idx][i]) + epsilon);
            }
        }
    }

    void zero_grad() {
        for (auto& p : parameters) {
            p->zero_grad();
        }
    }
};

#endif