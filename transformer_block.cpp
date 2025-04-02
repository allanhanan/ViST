#include "multi_head_attention.hpp"
#include "feedforward.hpp"  
#include "utils.hpp" 

class TransformerBlock {
private:
    int embedding_dim;
    int num_heads;
    int d_k, d_v;
    int hidden_dim;  // For FFN (typically 4x embedding_dim)

    // Learnable weights for FFN layers
    // W1: [embedding_dim, hidden_dim], W2: [hidden_dim, embedding_dim]
    std::vector<std::vector<float>> W1;
    std::vector<std::vector<float>> W2;
    std::vector<float> bias1;  // [hidden_dim]
    std::vector<float> bias2;  // [embedding_dim]

    // Weights for multi-head attention: each of shape [embedding_dim, embedding_dim]
    std::vector<std::vector<float>> Wq;
    std::vector<std::vector<float>> Wk;
    std::vector<std::vector<float>> Wv;
    std::vector<float> bias_q; // [embedding_dim]
    std::vector<float> bias_k; // [embedding_dim]
    std::vector<float> bias_v; // [embedding_dim]

    // Layer Norm parameters
    std::vector<float> gamma;  // [embedding_dim]
    std::vector<float> beta;   // [embedding_dim]

public:
    TransformerBlock(int embed_dim, int heads)
        : embedding_dim(embed_dim), num_heads(heads),
          d_k(embed_dim), d_v(embed_dim),  // Force d_k = d_v = embedding_dim for residual connection compatibility
          hidden_dim(4 * embed_dim),
          gamma(embed_dim, 1.0f),     // Scale parameter for layer normalization in FFN
          beta(embed_dim, 0.0f) {     // Bias parameter for layer normalization in FFN

        // Initialize FFN weights:
        // W1: [embedding_dim, hidden_dim], W2: [hidden_dim, embedding_dim]
        W1 = random_matrix(embedding_dim, hidden_dim);
        W2 = random_matrix(hidden_dim, embedding_dim);
        bias1 = random_vector(hidden_dim);
        bias2 = random_vector(embedding_dim);

        // Initialize attention weights:
        // Each of shape [embedding_dim, embedding_dim]
        Wq = random_matrix(embedding_dim, d_k);
        Wk = random_matrix(embedding_dim, d_k);
        Wv = random_matrix(embedding_dim, d_v);
        bias_q = random_vector(d_k);
        bias_k = random_vector(d_k);
        bias_v = random_vector(d_v);
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input) {
        // input: [seq_length, embedding_dim] (where seq_length = number of patches)

        // Multi-Head Self-Attention
        auto attention_output = multi_head_attention(input, Wq, Wk, Wv,
                                                     bias_q, bias_k, bias_v,
                                                     num_heads, d_k, d_v);
        // Residual connection + Layer Norm
        auto norm1_input = add_matrices(input, attention_output);
        auto norm1_output = layer_norm(norm1_input, gamma, beta);

        // Feedforward Network (using separate module)
        auto ff_output = feedforward_network(norm1_output, W1, bias1, W2, bias2);

        // Residual connection + Layer Norm after FFN
        auto norm2_input = add_matrices(norm1_output, ff_output);
        auto norm2_output = layer_norm(norm2_input, gamma, beta);

        return norm2_output;  // [seq_length, embedding_dim]
    }
};
