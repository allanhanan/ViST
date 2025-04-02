#ifndef MULTI_HEAD_ATTENTION_HPP
#define MULTI_HEAD_ATTENTION_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "utils.hpp"

inline std::vector<std::vector<float>> multi_head_attention(
    const std::vector<std::vector<float>>& input,  // Input sequence with shape [seq_length, embedding_dim]
    const std::vector<std::vector<float>>& Wq,       // Query weight matrix with shape [embedding_dim, d_k]
    const std::vector<std::vector<float>>& Wk,       // Key weight matrix with shape [embedding_dim, d_k]
    const std::vector<std::vector<float>>& Wv,       // Value weight matrix with shape [embedding_dim, d_v]
    const std::vector<float>& bias_q,                // Query bias vector with shape [d_k]
    const std::vector<float>& bias_k,                // Key bias vector with shape [d_k]
    const std::vector<float>& bias_v,                // Value bias vector with shape [d_v]
    int num_heads,                                   // Number of attention heads
    int d_k,                                         // Total dimension for Query and Key (we now set d_k = embedding_dim)
    int d_v                                          // Total dimension for Value (we now set d_v = embedding_dim)
) {
    int seq_length = input.size();  // Get the sequence length from the input
    int embedding_dim = input[0].size();  // Get the embedding dimension from the input
    
    std::cout << "[LOG] Starting multi-head attention..." << std::endl;
    
    // Step 1: Compute linear projections; result shape: [seq_length, d_*]
    std::cout << "[LOG] Computing linear projections for Q, K, and V..." << std::endl;
    
    auto Q = linear_transform(input, Wq, bias_q);  // Compute Query embeddings
    auto K = linear_transform(input, Wk, bias_k);  // Compute Key embeddings
    auto V = linear_transform(input, Wv, bias_v);  // Compute Value embeddings
    
    std::cout << "[LOG] Linear projections done." << std::endl;
    
    // Prepare storage for the outputs of each attention head
    std::vector<std::vector<std::vector<float>>> heads_output;
    
    int head_dim = d_k / num_heads;  // Calculate dimension per head (since d_k = embedding_dim, head_dim = embedding_dim/num_heads)
    std::cout << "[LOG] Processing " << num_heads << " attention heads..." << std::endl;
    
    for (int head = 0; head < num_heads; ++head) {
        std::cout << "[LOG] Processing head " << (head + 1) << " of " << num_heads << "..." << std::endl;
        
        int start_idx = head * head_dim;  // Calculate the starting index for this head
        int end_idx = (head + 1) * head_dim;  // Calculate the ending index for this head
        
        // Allocate memory for the slices of Q, K, and V for this head
        std::vector<std::vector<float>> Q_head(seq_length, std::vector<float>(head_dim));
        std::vector<std::vector<float>> K_head(seq_length, std::vector<float>(head_dim));
        std::vector<std::vector<float>> V_head(seq_length, std::vector<float>(head_dim));  // d_v == d_k now
        
        for (int i = 0; i < seq_length; ++i) {
            for (int j = start_idx; j < end_idx; ++j) {
                Q_head[i][j - start_idx] = Q[i][j];  // Slice the Query embeddings
                K_head[i][j - start_idx] = K[i][j];  // Slice the Key embeddings
                V_head[i][j - start_idx] = V[i][j];  // Slice the Value embeddings
            }
        }
        
        std::cout << "[LOG] Head sliced." << std::endl;
        
        // Compute attention scores for this head
        auto scores = matmul(Q_head, transpose(K_head));  // Matrix multiplication of Query and transposed Key to get scores
        
        std::vector<std::vector<float>> attention_scores(seq_length, std::vector<float>(head_dim));
        for (int i = 0; i < seq_length; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                attention_scores[i][j] = scores[i][j] / sqrt(d_k);  // Scale the scores by the square root of d_k
                attention_scores[i][j] = std::max(attention_scores[i][j], 1e-6f);  // Avoid division by zero by addimg a small value episilon
            }
        }
        
        std::cout << "[LOG] Attention scores computed." << std::endl;
        
        // Compute the weighted sum of Value embeddings based on attention scores
        auto values = matmul(attention_scores, V_head);  // Matrix multiplication of attention scores and Value embeddings
        
        heads_output.push_back(values);  // Store the output of this head
    }
    
    std::cout << "[LOG] Heads processed." << std::endl;
    
    // Concatenate the outputs from all attention heads
    std::vector<std::vector<float>> concatenated(seq_length, std::vector<float>());
    
    for (int i = 0; i < seq_length; ++i) {
        for (int head = 0; head < num_heads; ++head) {
            concatenated[i].insert(concatenated[i].end(),
                                   heads_output[head][i].begin(),
                                   heads_output[head][i].end());
        }
    }
    
    std::cout << "[LOG] Heads concatenated." << std::endl;
    
    // Apply a final linear transformation to the concatenated output
    std::cout << "[LOG] Applying final projection..." << std::endl;
    
    auto final_output = linear_transform(concatenated, Wq, bias_q);  // Output shape: [seq_length, d_k]
    
    std::cout << "[LOG] Final projection done." << std::endl;
    
    return final_output;
}
#endif // MULTI_HEAD_ATTENTION_HPP