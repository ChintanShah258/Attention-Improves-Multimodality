import torch
from torch import nn

class GatingMechanismTrainer:
    def __init__(self, model, optimizer, scheduler=None, device="cuda", lambda_reconstruction=0.5, lambda_alignment=0.5, margin=0.5):
        """
        Initialize the GatingMechanismTrainer.

        Args:
            model (torch.nn.Module): The GatingMechanism model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            device (str): Device to use for training ('cuda' or 'cpu').
            lambda_reconstruction (float): Weight for the reconstruction loss.
            lambda_alignment (float): Weight for the alignment loss.
            margin (float): Margin for contrastive loss.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_alignment = lambda_alignment
        self.margin = margin

        # Loss functions
        self.reconstruction_loss_fn = nn.MSELoss()

    def compute_combined_loss(self, enhanced_features, original_image_features, text_features):
        """
        Compute the combined loss (reconstruction + contrastive alignment using weighted average).
        """
        # Normalize features for cosine similarity
        enhanced_features = nn.functional.normalize(enhanced_features, p=2, dim=1)

        # Compute similarity matrix: (N, M) where N=batch size, M=number of text features
        similarity_matrix = torch.matmul(enhanced_features, text_features.T)  # Shape: [N, M]

        # Compute softmax weights for weighted average
        similarity_weights = nn.functional.softmax(similarity_matrix, dim=1)  # Shape: [N, M]

        # Compute weighted average of text features for each image
        weighted_text_features = torch.matmul(similarity_weights, text_features)  # Shape: [N, feature_dim]

        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(enhanced_features, original_image_features)

        # Cosine similarity between enhanced features and weighted text features
        weighted_similarity = torch.sum(enhanced_features * weighted_text_features, dim=1)  # Shape: [N]

        # Contrastive loss
        contrastive_loss = torch.mean(1 - weighted_similarity)

        # Combined loss
        combined_loss = (self.lambda_reconstruction * reconstruction_loss +
                        self.lambda_alignment * contrastive_loss)

        return combined_loss

    def train(self, image_features, text_features):
        """
        Train the gating mechanism for a single batch.

        Args:
            image_features (torch.Tensor): Features from the image encoder.
            text_features (torch.Tensor): Precomputed text features.

        Returns:
            torch.Tensor: Enhanced image features.
        """
        self.model.train()

        # Ensure input features have gradients
        image_features.requires_grad_(True)

        # Forward pass through the gating mechanism
        enhanced_features = self.model(image_features, text_features)

        # Ensure enhanced features require gradients
        #assert enhanced_features.requires_grad, "Enhanced features do not require gradients!"

        # Compute loss
        loss = self.compute_combined_loss(enhanced_features, image_features, text_features)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()  # Ensure loss is part of the computation graph
        self.optimizer.step()

        # Step the scheduler if provided
        if self.scheduler:
            self.scheduler.step()

        return enhanced_features


    def freeze(self):
        """
        Freeze the weights of the GatingMechanism model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
