#include "../include/ops.h"
#include <cassert>
#include <algorithm>
#include <cmath>

TensorPtr matmul(TensorPtr A, TensorPtr B) {
    assert(A->cols == B->rows && "Dimensi MatMul Salah!");

    TensorPtr C = Tensor::create(A->rows, B->cols);
    C->prev = {A, B};

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; k++) {
                sum += A->at(i, k) * B->at(k, j);
            }
            C->at(i, j) = sum;
        }
    }

    C->_backward = [A, B, C]() {
        // Optimized: dA = dC @ B^T, dB = A^T @ dC
        for (int i = 0; i < A->rows; i++) {
            for (int k = 0; k < A->cols; k++) {
                float grad_a = 0.0f;
                for (int j = 0; j < B->cols; j++) {
                    grad_a += C->grad_at(i, j) * B->at(k, j);
                }
                A->grad_at(i, k) += grad_a;
            }
        }

        for (int k = 0; k < B->rows; k++) {
            for (int j = 0; j < B->cols; j++) {
                float grad_b = 0.0f;
                for (int i = 0; i < A->rows; i++) {
                    grad_b += A->at(i, k) * C->grad_at(i, j);
                }
                B->grad_at(k, j) += grad_b;
            }
        }
    };

    return C;
}

TensorPtr relu(TensorPtr input) {
    TensorPtr output = Tensor::create(input->rows, input->cols);
    output->prev = {input};

    for (size_t i = 0; i < input->data.size(); i++) {
        output->data[i] = std::max(0.0f, input->data[i]);
    }

    output->_backward = [input, output]() {
        for (size_t i = 0; i < input->data.size(); i++) {
            if (input->data[i] > 0) {
                input->grad[i] += output->grad[i];
            } else {
                // Gradien 0, do nothing
            }
        }
    };

    return output;
}

TensorPtr sub(TensorPtr A, TensorPtr B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    TensorPtr C = Tensor::create(A->rows, A->cols);
    C->prev = {A, B};

    // Forward: C = A - B
    for (size_t i = 0; i < A->data.size(); i++) {
        C->data[i] = A->data[i] - B->data[i];
    }

    // Backward
    C->_backward = [A, B, C]() {
        for (size_t i = 0; i < A->data.size(); i ++) {
            A->grad[i] += C->grad[i];
            B->grad[i] -= C->grad[i];
        }
    };

    return C;
}

TensorPtr mse_loss(TensorPtr pred, TensorPtr target) {
    assert(pred->rows == target->rows && pred->cols == target->cols);

    /**
     * Output loss is usually scalar 1x1, but we keep the size the same
     * for simplicity.
     * Actually, MSE averages the error. Here we'll use Sum Squared Error
     * first for simplicity.
     */
    TensorPtr loss = Tensor::create(1, 1);
    loss->prev = {pred, target};

    // forward: sum((pred-target)^2)
    float sum_sq_error = 0.0f;
    for (size_t i = 0; i < pred->data.size(); i++) {
        float diff = pred->data[i] - target->data[i];
        sum_sq_error += diff * diff;
    }
    loss->data[0] = sum_sq_error / pred->data.size();

    loss->_backward = [pred, target, loss]() {
        float n = (float)pred->data.size();
        for (size_t i = 0; i < pred->data.size(); i++) {
            float diff = pred->data[i] - target->data[i];
            float grad = (2.0f * diff / n) * loss->grad[0];
            pred->grad[i] += grad;
        }
    };

    return loss;
}

TensorPtr transpose(TensorPtr A) {
    TensorPtr C = Tensor::create(A->cols, A->rows);
    C->prev = {A};

    // Forward: C[j, i] = A[i, j]
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->at(j, i) = A->at(i, j);
        }
    }

    // Backward: Grad A[i, j] += Grad C[j, i]
    C->_backward = [A, C]() {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < A->cols; j++) {
                A->grad_at(i, j) += C->grad_at(j, i);
            }
        }
    };
    return C;
}

TensorPtr softmax(TensorPtr input) {
    TensorPtr output = Tensor::create(input->rows, input->cols);
    output->prev = {input};

    // Forward (Row-wise Softmax)
    for (int i = 0; i < input->rows; i++) {
        float max_val = -1e9;
        for (int j = 0; j < input->cols; j++) max_val = std::max(max_val, input->at(i, j));

        float sum_exp = 0.0f;
        for (int j = 0; j < input->cols; j++) {
            float val = std::exp(input->at(i, j) - max_val);
            output->at(i, j) = val;
            sum_exp += val;
        }

        for (int j = 0; j < input->cols; j++) {
            output->at(i, j) /= sum_exp;
        }
    }

    output->_backward = [input, output]() {
        for (int i = 0; i < input->rows; i++) {
            float dot = 0.0f;
            for (int k = 0; k < input->cols; k++) {
                dot += output->at(i, k) * output->grad_at(i, k);
            }

            for (int j = 0; j < input->cols; j++) {
                float s = output->at(i, j);
                float g = output->grad_at(i, j);
                input->grad_at(i, j) += s * (g - dot);
            }
        }
    };
    return output;
}

// === ADDITIONAL OPERATIONS ===

TensorPtr add(TensorPtr A, TensorPtr B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    TensorPtr C = Tensor::create(A->rows, A->cols);
    C->prev = {A, B};

    // Forward: C = A + B
    for (size_t i = 0; i < A->data.size(); i++) {
        C->data[i] = A->data[i] + B->data[i];
    }

    // Backward: dA = dC, dB = dC
    C->_backward = [A, B, C]() {
        for (size_t i = 0; i < A->data.size(); i++) {
            A->grad[i] += C->grad[i];
            B->grad[i] += C->grad[i];
        }
    };

    return C;
}

TensorPtr multiply(TensorPtr A, TensorPtr B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    TensorPtr C = Tensor::create(A->rows, A->cols);
    C->prev = {A, B};

    // Forward: C = A * B (element-wise)
    for (size_t i = 0; i < A->data.size(); i++) {
        C->data[i] = A->data[i] * B->data[i];
    }

    // Backward: dA = dC * B, dB = dC * A
    C->_backward = [A, B, C]() {
        for (size_t i = 0; i < A->data.size(); i++) {
            A->grad[i] += C->grad[i] * B->data[i];
            B->grad[i] += C->grad[i] * A->data[i];
        }
    };

    return C;
}

TensorPtr tanh_activation(TensorPtr input) {
    TensorPtr output = Tensor::create(input->rows, input->cols);
    output->prev = {input};

    // Forward: tanh(x)
    for (size_t i = 0; i < input->data.size(); i++) {
        output->data[i] = std::tanh(input->data[i]);
    }

    // Backward: d_tanh = (1 - tanh^2) * grad_out
    output->_backward = [input, output]() {
        for (size_t i = 0; i < input->data.size(); i++) {
            float tanh_val = output->data[i];
            input->grad[i] += (1.0f - tanh_val * tanh_val) * output->grad[i];
        }
    };

    return output;
}

TensorPtr sigmoid(TensorPtr input) {
    TensorPtr output = Tensor::create(input->rows, input->cols);
    output->prev = {input};

    // Forward: sigmoid(x) = 1 / (1 + exp(-x))
    for (size_t i = 0; i < input->data.size(); i++) {
        output->data[i] = 1.0f / (1.0f + std::exp(-input->data[i]));
    }

    // Backward: d_sigmoid = sigmoid * (1 - sigmoid) * grad_out
    output->_backward = [input, output]() {
        for (size_t i = 0; i < input->data.size(); i++) {
            float sig_val = output->data[i];
            input->grad[i] += sig_val * (1.0f - sig_val) * output->grad[i];
        }
    };

    return output;
}

TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target) {
    assert(pred->rows == target->rows && pred->cols == target->cols);

    TensorPtr loss = Tensor::create(1, 1);
    loss->prev = {pred, target};

    // Forward: -sum(target * log(pred + eps)) / batch_size
    float total_loss = 0.0f;
    const float eps = 1e-7f; // For numerical stability

    for (size_t i = 0; i < pred->data.size(); i++) {
        total_loss -= target->data[i] * std::log(pred->data[i] + eps);
    }
    loss->data[0] = total_loss / pred->rows;

    // Backward: -target / (pred + eps) * grad_loss / batch_size
    loss->_backward = [pred, target, loss]() {
        const float eps = 1e-7f;
        float n = (float)pred->rows;

        for (size_t i = 0; i < pred->data.size(); i++) {
            pred->grad[i] += (-target->data[i] / (pred->data[i] + eps)) * loss->grad[0] / n;
        }
    };

    return loss;
}