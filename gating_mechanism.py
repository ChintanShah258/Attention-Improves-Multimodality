import torch
from torch import nn

class GatingMechanism(nn.Module):
    def __init__(self, image_feature_dim=1024, text_feature_dim=1024, projection_dim=1024, top_k=5):
        super(GatingMechanism, self).__init__()
        self.projection_dim = projection_dim
        self.top_k = top_k
        self.image_proj = nn.Linear(image_feature_dim, self.projection_dim, bias=False)
        self.text_proj = nn.Linear(text_feature_dim, self.projection_dim, bias=False)
        self.final_proj = nn.Linear(self.projection_dim, image_feature_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, image_features, text_features):
    # Ensure inputs require gradients
        #assert image_features.requires_grad, "image_features does not require gradients!"
        #assert text_features.requires_grad, "text_features does not require gradients!"

        # Project image and text features
        image_features_proj = self.image_proj(image_features)  # Shape: (batch_size, projection_dim)
        text_features_proj = self.text_proj(text_features)    # Shape: (num_texts, projection_dim)

        batch_size = image_features_proj.size(0)
        enhanced_features_list = []

        for i in range(batch_size):
            current_image_feature = image_features_proj[i:i+1]  # Shape: (1, projection_dim)

            # Compute attention scores
            attention_scores = torch.matmul(current_image_feature, text_features_proj.T)  # Shape: (1, num_texts)

            # Select top-k text features
            top_k_values, top_k_indices = torch.topk(attention_scores, self.top_k, dim=1)  # Shape: (1, top_k)
            top_k_text_features = torch.index_select(text_features_proj, dim=0, index=top_k_indices.view(-1))  # Shape: (top_k, projection_dim)

            # Compute weighted interactions
            top_k_weights = torch.softmax(top_k_values, dim=1).unsqueeze(-1)  # Shape: (1, top_k, 1)
            weighted_interactions = current_image_feature.unsqueeze(1) * top_k_weights * top_k_text_features.unsqueeze(0)  # Shape: (1, top_k, projection_dim)

            # Sum over top-k interactions
            enhanced_feature = weighted_interactions.sum(dim=1)  # Shape: (1, projection_dim)

            enhanced_features_list.append(enhanced_feature)

        # Combine enhanced features for all images
        enhanced_features = torch.cat(enhanced_features_list, dim=0)  # Shape: (batch_size, projection_dim)

        # Final projection and activation
        enhanced_features = self.final_proj(enhanced_features)
        enhanced_features = self.activation(enhanced_features)

        # Ensure the output requires gradients
        #assert enhanced_features.requires_grad, "Output does not require gradients!"
        return enhanced_features




