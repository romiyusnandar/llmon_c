#include "../include/ops.h"
#include <cassert>
#include <algorithm>

Tensor matmul(Tensor& A, Tensor& B) {
    assert(A.cols == B.rows && "Dimensi MatMul Salah!");

    Tensor C(A.rows, B.cols);

    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
    return C;
}

Tensor relu(Tensor& input) {
    Tensor output(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++) {
        output.data[i] = std::max(0.0f, input.data[i]);
    }

    return output;
}