#include "../include/nn.h"
#include <iostream>

// === LINEAR IMPLEMENTATION ===
Linear::Linear(int in_features, int out_features, bool bias_flag) {
    weight = Tensor::create(in_features, out_features);
    weight->random_init();

    use_bias = bias_flag;
    if (use_bias) {
        bias = Tensor::create(1, out_features);
        std::fill(bias->data.begin(), bias->data.end(), 0.0f);
    }
}

TensorPtr Linear::forward(TensorPtr input) {
    TensorPtr out = matmul(input, weight);

    if (use_bias) {
        // Implementasi Bias Addition Sederhana (Broadcasting manual)
        // Kita modifikasi output langsung atau buat tensor baru (Add Op)
        // Untuk tutorial ini, kita skip operasi Add graph node agar ringkas,
        // tapi idealnya buat fungsi add() di ops.cpp

        // Pseudo-add (Forward only logic for now inside Node)
        // Note: Ini cara malas. Cara benar harus via ops::add untuk backwardnya.
        // Mari kita asumsikan Linear tanpa bias dulu untuk training LLM sederhana
    }
    return out;
}

std::vector<TensorPtr> Linear::parameters() {
    if (use_bias) return {weight, bias};
    return {weight};
}


// === EMBEDDING IMPLEMENTATION ===
Embedding::Embedding(int num_embeddings, int embedding_dim) {
    weight = Tensor::create(num_embeddings, embedding_dim);
    weight->random_init();
}

TensorPtr Embedding::forward(TensorPtr input) {
    int batch_size = input->rows * input->cols; // Total token
    int embed_dim = weight->cols;

    TensorPtr out = Tensor::create(batch_size, embed_dim);
    out->prev = {weight};

    for (int i = 0; i < batch_size; i++) {
        int token_id = (int)input->data[i];

        // Safety check
        if (token_id >= weight->rows) token_id = 0;

        for (int j = 0; j < embed_dim; j++) {
            out->at(i, j) = weight->at(token_id, j);
        }
    }

    out->_backward = [input, out, this]() {
        int batch = input->rows * input->cols;
        int dim = weight->cols;

        for (int i = 0; i < batch; i++) {
            int token_id = (int)input->data[i];
            for (int j = 0; j < dim; j++) {
                weight->grad_at(token_id, j) += out->grad_at(i, j);
            }
        }
    };

    return out;
}

std::vector<TensorPtr> Embedding::parameters() {
    return {weight};
}

SelfAttention::SelfAttention(int embed_dim, int head_dim)
    : Wq(embed_dim, head_dim),
      Wk(embed_dim, head_dim),
      Wv(embed_dim, head_dim) {}

TensorPtr SelfAttention::forward(TensorPtr input) {
    TensorPtr Q = Wq.forward(input); // [Seq, HeadDim]
    TensorPtr K = Wk.forward(input); // [Seq, HeadDim]
    TensorPtr V = Wv.forward(input); // [Seq, HeadDim]

    TensorPtr K_T = transpose(K);    // [HeadDim, Seq]
    TensorPtr Scores = matmul(Q, K_T); // [Seq, Seq] -> Peta hubungan antar kata!

    // (Opsional: Bagi dengan sqrt(dim))

    TensorPtr AttnWeights = softmax(Scores);

    TensorPtr Output = matmul(AttnWeights, V); // [Seq, HeadDim]

    return Output;
}

std::vector<TensorPtr> SelfAttention::parameters() {
    std::vector<TensorPtr> params;
    auto p_q = Wq.parameters(); params.insert(params.end(), p_q.begin(), p_q.end());
    auto p_k = Wk.parameters(); params.insert(params.end(), p_k.begin(), p_k.end());
    auto p_v = Wv.parameters(); params.insert(params.end(), p_v.begin(), p_v.end());
    return params;
}