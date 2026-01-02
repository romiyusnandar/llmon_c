#include "../include/ops.h"
#include <cassert>
#include <algorithm>

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
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < B->cols; j++) {
                float grad_val = C->grad_at(i, j);

                for (int k = 0; k < A->cols; k++) {
                    A->grad_at(i, k) += grad_val * B->at(k, j);
                }

                for (int k = 0; k < A->cols; k++) {
                    B->grad_at(k, j) += A->at(i, k) * grad_val;
                }
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