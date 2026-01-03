# Quick Reference - API Documentation

## Core Classes

### Tensor
```cpp
// Creation
auto t = Tensor::create(rows, cols);

// Data access
t->at(i, j) = value;
float val = t->at(i, j);

// Gradients
t->grad_at(i, j) = grad_value;
t->zero_grad();  // Reset gradients

// Backward pass
t->backward();  // Compute gradients for entire graph

// Utilities
t->print();
t->print_grad();
float m = t->mean();
float s = t->std_dev();
```

### Operations (ops.h)

#### Matrix Operations
```cpp
TensorPtr matmul(TensorPtr A, TensorPtr B);
TensorPtr transpose(TensorPtr A);
```

#### Element-wise Operations
```cpp
TensorPtr add(TensorPtr A, TensorPtr B);
TensorPtr sub(TensorPtr A, TensorPtr B);
TensorPtr multiply(TensorPtr A, TensorPtr B);
```

#### Activations
```cpp
TensorPtr relu(TensorPtr input);
TensorPtr tanh_activation(TensorPtr input);
TensorPtr sigmoid(TensorPtr input);
TensorPtr softmax(TensorPtr input);  // Row-wise
```

#### Loss Functions
```cpp
TensorPtr mse_loss(TensorPtr pred, TensorPtr target);
TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target);
```

### Neural Network Layers

#### Linear Layer
```cpp
Linear layer(in_features, out_features, use_bias);
auto output = layer.forward(input);
auto params = layer.parameters();  // {weight, bias}
```

#### Embedding Layer
```cpp
Embedding embed(vocab_size, embed_dim);
auto output = embed.forward(token_ids);
// input: [seq_len, 1] with token IDs
// output: [seq_len, embed_dim]
```

#### Positional Embedding
```cpp
PositionalEmbedding pos_embed(max_seq_len, embed_dim);
auto output = pos_embed.forward(input);
// Adds positional information: output = input + pos_weight
```

#### Self-Attention
```cpp
SelfAttention attn(embed_dim, head_dim);
auto output = attn.forward(input);
// Applies scaled dot-product attention
```

#### Transformer Block
```cpp
TransformerBlock block(embed_dim, head_dim);
auto output = block.forward(input);
// Self-Attention + FFN + ReLU
```

#### GPT Model
```cpp
GPT model(vocab_size, embed_dim, max_seq_len, head_dim);
auto logits = model.forward(input);
// Complete architecture:
// Token Embed → Pos Embed → Transformer → Output Head
```

### Optimizers

#### SGD
```cpp
SGD optimizer(model.parameters(), learning_rate);

// Training step
optimizer.zero_grad();
loss->backward();
optimizer.step();
```

#### Adam
```cpp
Adam optimizer(model.parameters(), learning_rate, beta1, beta2, epsilon);
// Default: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

optimizer.zero_grad();
loss->backward();
optimizer.step();
```

## Common Patterns

### Training Loop
```cpp
for (int epoch = 0; epoch < epochs; epoch++) {
    // Forward pass
    auto output = model.forward(input);
    auto loss = cross_entropy_loss(output, target);

    // Backward pass
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();

    // Logging
    if (epoch % 100 == 0) {
        std::cout << "Epoch " << epoch << " Loss: " << loss->data[0] << "\n";
    }
}
```

### Inference
```cpp
// Get predictions
auto logits = model.forward(input);
auto probs = softmax(logits);

// Find argmax
int predicted = 0;
float max_prob = probs->at(0, 0);
for (int i = 1; i < vocab_size; i++) {
    if (probs->at(0, i) > max_prob) {
        max_prob = probs->at(0, i);
        predicted = i;
    }
}
```

### Creating Target (One-Hot)
```cpp
auto target = Tensor::create(seq_len, vocab_size);
std::fill(target->data.begin(), target->data.end(), 0.0f);

for (int i = 0; i < seq_len; i++) {
    int target_token = labels[i];
    target->at(i, target_token) = 1.0f;
}
```

## Debugging Tips

### Check Gradients
```cpp
// After backward()
weight->print_grad();

// Check for NaN
for (auto& g : weight->grad) {
    if (std::isnan(g)) {
        std::cout << "NaN gradient detected!\n";
    }
}
```

### Monitor Loss
```cpp
std::cout << "Loss: " << loss->data[0] << "\n";

// Loss should decrease
// If stuck: lower learning rate
// If exploding: check data normalization
```

### Inspect Activations
```cpp
auto hidden = layer.forward(input);
std::cout << "Mean: " << hidden->mean() << "\n";
std::cout << "Std: " << hidden->std_dev() << "\n";

// Healthy values: mean ≈ 0, std ≈ 1
```

### Parameter Count
```cpp
int total_params = 0;
for (auto& p : model.parameters()) {
    total_params += p->data.size();
}
std::cout << "Total parameters: " << total_params << "\n";
```

## Performance Tips

### 1. Learning Rate Selection
```cpp
// Start with Adam default
Adam opt(params, 0.001f);

// For simple tasks, can increase
Adam opt(params, 0.01f);

// For complex tasks, decrease
Adam opt(params, 0.0001f);
```

### 2. Batch Training
```cpp
// Process multiple sequences together
// Concatenate inputs vertically
auto batch_input = Tensor::create(batch_size * seq_len, 1);
// Fill with data...
```

### 3. Gradient Clipping (Manual)
```cpp
void clip_gradients(std::vector<TensorPtr>& params, float max_norm) {
    for (auto& p : params) {
        for (auto& g : p->grad) {
            if (g > max_norm) g = max_norm;
            if (g < -max_norm) g = -max_norm;
        }
    }
}
```

## Examples

### Example 1: Simple Linear Regression
```cpp
auto input = Tensor::create(3, 1);
input->data = {1.0f, 2.0f, 3.0f};

auto target = Tensor::create(3, 1);
target->data = {2.0f, 4.0f, 6.0f};  // y = 2x

auto weight = Tensor::create(1, 1);
weight->random_init();

SGD opt({weight}, 0.1f);

for (int i = 0; i < 100; i++) {
    auto pred = matmul(input, weight);
    auto loss = mse_loss(pred, target);
    opt.zero_grad();
    loss->backward();
    opt.step();
}
// weight->data[0] ≈ 2.0
```

### Example 2: Classification
```cpp
Linear classifier(input_dim, num_classes, true);
Adam opt(classifier.parameters(), 0.01f);

for (int epoch = 0; epoch < epochs; epoch++) {
    auto logits = classifier.forward(input);
    auto probs = softmax(logits);
    auto loss = cross_entropy_loss(probs, one_hot_target);

    opt.zero_grad();
    loss->backward();
    opt.step();
}
```

### Example 3: Sequence Prediction
```cpp
GPT model(vocab_size, 16, 10, 16);
Adam opt(model.parameters(), 0.01f);

// Train
for (int epoch = 0; epoch < 500; epoch++) {
    auto logits = model.forward(input_sequence);
    auto probs = softmax(logits);
    auto loss = cross_entropy_loss(probs, target_sequence);

    opt.zero_grad();
    loss->backward();
    opt.step();
}

// Generate
auto next_token_logits = model.forward(context);
auto next_token_probs = softmax(next_token_logits);
// Sample or take argmax
```

## Error Messages

### "Dimensi MatMul Salah!"
- Check: `A->cols == B->rows`
- Matrix multiplication requires compatible dimensions

### NaN in loss
- Check data normalization
- Lower learning rate
- Check for division by zero

### Memory issues
- Large sequences: use batching
- Deep models: implement gradient checkpointing

---

**For more details, see ARCHITECTURE.md**
