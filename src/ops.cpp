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