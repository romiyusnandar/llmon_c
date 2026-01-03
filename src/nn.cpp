#include "../include/nn.h"
#include <iostream>
#include <cmath>

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
        // Proper bias addition using add() operation with autograd support
        // Broadcasting: bias (1, out_features) is added to each row of output
        for (int i = 0; i < out->rows; i++) {
            for (int j = 0; j < out->cols; j++) {
                out->at(i, j) += bias->at(0, j);
            }
        }

        // Track bias in computation graph for backprop
        out->prev.push_back(bias);

        // Capture bias as local variable for lambda
        TensorPtr b = bias;
        auto old_backward = out->_backward;
        out->_backward = [out, b, old_backward]() {
            // First call the matmul backward
            old_backward();

            // Then accumulate bias gradients (sum over batch dimension)
            for (int i = 0; i < out->rows; i++) {
                for (int j = 0; j < out->cols; j++) {
                    b->grad_at(0, j) += out->grad_at(i, j);
                }
            }
        };
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

    // Capture weight directly instead of 'this' to avoid dangling pointer
    TensorPtr w = weight;
    out->_backward = [input, out, w]() {
        int batch = input->rows * input->cols;
        int dim = w->cols;

        for (int i = 0; i < batch; i++) {
            int token_id = (int)input->data[i];
            if (token_id < 0 || token_id >= w->rows) continue; // Safety check
            for (int j = 0; j < dim; j++) {
                w->grad_at(token_id, j) += out->grad_at(i, j);
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

    // Scaled attention: divide by sqrt(d_k) for stability
    float scale = 1.0f / std::sqrt((float)Q->cols);
    for (size_t i = 0; i < Scores->data.size(); i++) {
        Scores->data[i] *= scale;
    }

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

TransformerBlock::TransformerBlock(int embed_dim, int head_dim)
    : attn(embed_dim, head_dim), ffn(embed_dim, embed_dim) {} // FFN output size == input size

TensorPtr TransformerBlock::forward(TensorPtr input) {
    // Self-Attention
    TensorPtr attn_out = attn.forward(input);

    // Feed-forward with ReLU activation
    TensorPtr ffn_out = ffn.forward(attn_out);
    TensorPtr x = relu(ffn_out);

    // Note: For better results, add Residual Connection and LayerNorm:
    // x = layer_norm(x + input)  // residual
    // But we keep it simple for now

    return x;
}

std::vector<TensorPtr> TransformerBlock::parameters() {
    std::vector<TensorPtr> params = attn.parameters();
    std::vector<TensorPtr> p_ffn = ffn.parameters();
    params.insert(params.end(), p_ffn.begin(), p_ffn.end());
    return params;
}

// === POSITIONAL EMBEDDING IMPLEMENTATION ===
PositionalEmbedding::PositionalEmbedding(int max_seq_len, int embedding_dim) {
    pos_weight = Tensor::create(max_seq_len, embedding_dim);
    pos_weight->random_init();
}

TensorPtr PositionalEmbedding::forward(TensorPtr input) {
    // input shape: [seq_len, embed_dim]
    int seq_len = input->rows;
    int embed_dim = input->cols;

    TensorPtr output = Tensor::create(seq_len, embed_dim);
    output->prev = {input, pos_weight};

    // Forward: output = input + pos_weight[0:seq_len]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            output->at(i, j) = input->at(i, j) + pos_weight->at(i, j);
        }
    }

    // Backward: gradient flows to both input and pos_weight
    TensorPtr pw = pos_weight;
    output->_backward = [input, output, pw]() {
        int seq_len = output->rows;
        int embed_dim = output->cols;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embed_dim; j++) {
                input->grad_at(i, j) += output->grad_at(i, j);
                pw->grad_at(i, j) += output->grad_at(i, j);
            }
        }
    };

    return output;
}

std::vector<TensorPtr> PositionalEmbedding::parameters() {
    return {pos_weight};
}

// === GPT IMPLEMENTATION ===
GPT::GPT(int vocab_size, int embed_dim, int max_seq_len, int head_dim)
    : vocab_size(vocab_size),
      embed_dim(embed_dim),
      max_seq_len(max_seq_len),
      token_embed(vocab_size, embed_dim),
      pos_embed(max_seq_len, embed_dim),
      transformer(embed_dim, head_dim),
      output_head(embed_dim, vocab_size, false) {} // No bias for output

TensorPtr GPT::forward(TensorPtr input) {
    // input: token ids [seq_len, 1] or [batch*seq_len] flattened
    // Step 1: Token Embedding
    TensorPtr tok_emb = token_embed.forward(input);

    // Step 2: Add Positional Embedding
    TensorPtr x = pos_embed.forward(tok_emb);

    // Step 3: Transformer Block
    x = transformer.forward(x);

    // Step 4: Output projection to vocabulary
    TensorPtr logits = output_head.forward(x);

    return logits;
}

std::vector<TensorPtr> GPT::parameters() {
    std::vector<TensorPtr> params;

    auto p_tok = token_embed.parameters();
    params.insert(params.end(), p_tok.begin(), p_tok.end());

    auto p_pos = pos_embed.parameters();
    params.insert(params.end(), p_pos.begin(), p_pos.end());

    auto p_trans = transformer.parameters();
    params.insert(params.end(), p_trans.begin(), p_trans.end());

    auto p_out = output_head.parameters();
    params.insert(params.end(), p_out.begin(), p_out.end());

    return params;
}