import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# Channel Attention Module (Spatial)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()

        # Average pooling along time dimension
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)

        # Max pooling along time dimension
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)

        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1)

        return x * attention.expand_as(x), attention

# Temporal Attention Module
class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TemporalAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()

        # Average pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Apply temporal convolution
        temporal = self.conv(avg_out)
        attention = self.sigmoid(temporal)

        return x * attention.expand_as(x), attention

# Spectral Attention Module
class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpectralAttention, self).__init__()
        self.in_channels = in_channels  # Store for dimension checking

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add dimension check and fix
        b, c, f = x.size()  # f is frequency bins

        if c != self.in_channels:
            # Handle dimension mismatch by adaptive pooling if needed
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.in_channels).transpose(1, 2)
            _, c, _ = x.size()  # Update c after pooling

        # Global average pooling along frequency dimension
        y = torch.mean(x, dim=2)

        # Apply FC layers
        attention = self.fc(y).view(b, c, 1)

        return x * attention.expand_as(x), attention

# Enhanced Cascade Attention Module
class EnhancedCascadeAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_heads=4):
        super(EnhancedCascadeAttention, self).__init__()

        # First stage: Channel (Spatial) Attention
        self.channel_attention = ChannelAttention(in_channels)

        # Second stage: Temporal Attention
        self.temporal_attention = TemporalAttention()

        # Feature processing for each domain - ensure consistent dimensions
        self.spatial_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.temporal_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2)

        # FFT processing with specific output dimension
        self.fft_pool = nn.AdaptiveAvgPool1d(hidden_dim)

        # Third stage: Spectral Attention (for frequency domain) - match dimensions
        self.spectral_attention = SpectralAttention(hidden_dim)

        # Multi-head self-attention for feature interaction across domains
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Store dimensions for debugging
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, c, t = x.size()

        # Verify dimensions
        if c != self.in_channels:
            x = F.adaptive_avg_pool2d(x.unsqueeze(1), (self.in_channels, t)).squeeze(1)

        # Stage 1: Channel (Spatial) Attention
        spatial_out, spatial_weights = self.channel_attention(x)
        spatial_features = self.spatial_conv(spatial_out)  # Output: [B, hidden_dim, T]

        # Stage 2: Temporal Attention (cascaded on spatial output)
        temporal_out, temporal_weights = self.temporal_attention(spatial_out)
        temporal_features = self.temporal_conv(temporal_out)  # Output: [B, hidden_dim, T]

        # Stage 3: Spectral Attention (on frequency domain)
        x_fft = torch.fft.fft(spatial_out, dim=2).abs()
        x_fft = self.fft_pool(x_fft)  # Reshape to fixed dimensions: [B, C, hidden_dim]

        # Ensure spectral input has correct dimensions
        if x_fft.size(1) != self.hidden_dim:
            x_fft = x_fft.transpose(1, 2)  # Transpose if dimensions are swapped

        spectral_out, spectral_weights = self.spectral_attention(x_fft)

        # Pool features from each domain
        spatial_pooled = F.adaptive_avg_pool1d(spatial_features, 1).squeeze(-1)  # [B, hidden_dim]
        temporal_pooled = F.adaptive_avg_pool1d(temporal_features, 1).squeeze(-1)  # [B, hidden_dim]
        spectral_pooled = F.adaptive_avg_pool1d(spectral_out, 1).squeeze(-1)  # [B, hidden_dim]

        # Stack features for self-attention across domains
        stacked_features = torch.stack([
            spatial_pooled, temporal_pooled, spectral_pooled
        ], dim=1)  # Shape: [B, 3, hidden_dim]

        # Apply multi-head self attention across domains
        attended_features, _ = self.self_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Reshape back
        attended_spatial = attended_features[:, 0]
        attended_temporal = attended_features[:, 1]
        attended_spectral = attended_features[:, 2]

        # Concatenate attended features and apply fusion
        combined = torch.cat([
            attended_spatial,
            attended_temporal,
            attended_spectral
        ], dim=1)

        output = self.fusion(combined)

        return output, (spatial_weights, temporal_weights, spectral_weights)

# View-specific encoder with cascade attention
class EnhancedViewEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, dropout=0.3):
        super(EnhancedViewEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Initial feature extraction
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Cascade attention module with dimension matching
        self.cascade_attn = EnhancedCascadeAttention(hidden_dim, hidden_dim // 2)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Feature fusion
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Handle dimension and shape issues
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Ensure correct input channel dimension
        b, c, t = x.size()
        if c != self.in_channels:
            x = F.adaptive_avg_pool2d(x.unsqueeze(1), (self.in_channels, t)).squeeze(1)

        # Initial feature extraction
        x = self.conv1(x)  # Output: [B, hidden_dim, T]
        x = self.bn1(x)
        x = self.relu(x)

        try:
            # Apply cascade attention
            ca_features, _ = self.cascade_attn(x)

            # Global pooling
            x_avg = self.global_avg_pool(x).view(x.size(0), -1)
            x_max = self.global_max_pool(x).view(x.size(0), -1)

            # Combine pooled features with cascade attention features
            x_combined = torch.cat([
                torch.cat([x_avg, x_max], dim=1),  # Pooled features
                ca_features  # Cascade attention features
            ], dim=1)

            # Final projection
            x_final = self.fc(x_combined)

        except RuntimeError as e:
            print(f"Error in EnhancedViewEncoder forward: {str(e)}")
            print(f"Input shape: {x.shape}, Expected channels: {self.in_channels}")
            # Emergency fallback
            x_avg = self.global_avg_pool(x).view(x.size(0), -1)
            x_final = torch.zeros((x.size(0), self.hidden_dim), device=x.device)

        return x_final

# Integrated Fusion Module to combine multiple view representations
class IntegratedFusionModule(nn.Module):
    def __init__(self, feature_dim, num_views, hidden_dim):
        super(IntegratedFusionModule, self).__init__()

        # Self-attention for view interaction
        self.self_attention = nn.MultiheadAttention(feature_dim, 4, batch_first=True)

        # Cross-view gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Sigmoid()
        )

        # View-specific importance predictors
        self.importance_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Softplus()
            ) for _ in range(num_views)
        ])

        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Store dimensions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

    def forward(self, view_features):
        """
        view_features: List of view-specific features [B, feature_dim] for each view
        """
        if len(view_features) == 0:
            return None, None

        batch_size = view_features[0].size(0)
        feature_dim = view_features[0].size(1)

        # Stack features for attention
        stacked = torch.stack(view_features, dim=1)  # [B, num_views, feature_dim]

        # Apply self-attention for view interaction
        attended, _ = self.self_attention(stacked, stacked, stacked)

        # Calculate importance weights for each view
        importance_weights = []
        for idx, proj in enumerate(self.importance_projections):
            if idx < len(view_features):
                weight = proj(attended[:, idx])  # [B, 1]
                importance_weights.append(weight)

        # Normalize importance weights
        if importance_weights:
            importance_tensor = torch.cat(importance_weights, dim=1)  # [B, num_views]
            importance_weights = F.softmax(importance_tensor, dim=1)

            # Apply weighted combination
            weighted_sum = torch.zeros(batch_size, feature_dim).to(attended.device)
            for idx in range(len(view_features)):
                weighted_sum += attended[:, idx] * importance_weights[:, idx].unsqueeze(1)

            # Final fusion
            fused = self.final_fusion(weighted_sum)

            return fused, importance_weights
        else:
            # If no view features, return None
            return None, None

# Unified Multi-View EEG Encoder with Cascade Attention and Contrastive Learning
class UnifiedMultiViewEEGEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=Config.HIDDEN_DIM, dropout=0.3, projection_dim=Config.PROJECTION_DIM):
        super(UnifiedMultiViewEEGEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Main EEG encoder with enhanced cascade attention
        self.main_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Enhanced cascade attention for main path
        self.cascade_attention = EnhancedCascadeAttention(hidden_dim, hidden_dim)

        # View-specific encoders
        self.view_encoders = nn.ModuleDict({
            # Frequency band views
            'delta': EnhancedViewEncoder(in_channels, hidden_dim),
            'theta': EnhancedViewEncoder(in_channels, hidden_dim),
            'alpha': EnhancedViewEncoder(in_channels, hidden_dim),
            'beta': EnhancedViewEncoder(in_channels, hidden_dim),
            'gamma': EnhancedViewEncoder(in_channels, hidden_dim),

            # Time segment views
            'early': EnhancedViewEncoder(in_channels, hidden_dim),
            'mid': EnhancedViewEncoder(in_channels, hidden_dim),
            'late': EnhancedViewEncoder(in_channels, hidden_dim)
        })

        # Add spatial group encoders if we have enough channels
        if in_channels >= 64:
            frontal_channels = 16  # Approximate frontal channels
            central_channels = 16  # Approximate central channels
            parietal_channels = 16  # Approximate parietal channels
            occipital_channels = min(16, in_channels - 48)  # Approximate occipital channels

            self.view_encoders.update({
                'frontal': EnhancedViewEncoder(frontal_channels, hidden_dim),
                'central': EnhancedViewEncoder(central_channels, hidden_dim),
                'parietal': EnhancedViewEncoder(parietal_channels, hidden_dim),
                'occipital': EnhancedViewEncoder(occipital_channels, hidden_dim)
            })

        # Integrated fusion module for combining views
        self.view_fusion = IntegratedFusionModule(
            feature_dim=hidden_dim,
            num_views=len(self.view_encoders),
            hidden_dim=hidden_dim
        )

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Calculate expected combined dimensions
        # main_pooled = 2 * hidden_dim (avg + max pool)
        # ca_features = hidden_dim
        # view_representation = hidden_dim
        combined_dim = hidden_dim * 2 + hidden_dim + hidden_dim

        # Fusion of main and multi-view paths
        self.final_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Alternative branch for cases when view features aren't available
        self.alt_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, x, views=None, return_all=False):
        # Add dimension check
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Process the main path
        main_features = self.main_encoder(x)
        ca_features, _ = self.cascade_attention(main_features)

        # Global pooling for main path features
        main_avg = self.global_avg_pool(main_features).squeeze(-1)
        main_max = self.global_max_pool(main_features).squeeze(-1)
        main_pooled = torch.cat([main_avg, main_max], dim=1)

        # Process view-specific paths if views are provided
        view_features = []
        view_projections = {}
        view_importance = None

        if views is not None:
            for view_name, encoder in self.view_encoders.items():
                if view_name in views:
                    try:
                        # Get view-specific features
                        view_feat = encoder(views[view_name])
                        view_features.append(view_feat)

                        # Store projection for contrastive learning
                        view_projections[view_name] = self.projection(view_feat)
                    except Exception as e:
                        print(f"Error processing view {view_name}: {str(e)}")
                        # Skip this view on error
                        continue

            # Combine all view features with the integrated fusion module
            if view_features:
                try:
                    view_representation, view_importance = self.view_fusion(view_features)

                    # Combine main and view paths (if views were processed)
                    if view_representation is not None:
                        combined = torch.cat([
                            main_pooled,  # [B, hidden_dim*2]
                            ca_features,  # [B, hidden_dim]
                            view_representation  # [B, hidden_dim]
                        ], dim=1)

                        final_features = self.final_fusion(combined)
                    else:
                        # Fallback if view fusion failed
                        combined = torch.cat([main_pooled, ca_features], dim=1)
                        final_features = self.alt_fusion(combined)

                except Exception as e:
                    print(f"Error in view fusion: {str(e)}")
                    # Fallback if view fusion failed
                    combined = torch.cat([main_pooled, ca_features], dim=1)
                    final_features = self.alt_fusion(combined)
            else:
                # Fallback if no views were processed
                combined = torch.cat([main_pooled, ca_features], dim=1)
                final_features = self.alt_fusion(combined)
        else:
            # Fallback if no views were provided
            combined = torch.cat([main_pooled, ca_features], dim=1)
            final_features = self.alt_fusion(combined)

        if return_all:
            # Get global projection for contrastive learning
            global_proj = self.projection(final_features)

            # Return everything for training
            return final_features, global_proj, view_projections, view_importance
        else:
            # Return just the features for inference
            return final_features

# Unified Multi-View EEG Keyword Predictor with Label Embedding
class UnifiedEEGKeywordPredictor(nn.Module):
    def __init__(self, in_channels, num_labels, hidden_dim=Config.HIDDEN_DIM,
                 projection_dim=Config.PROJECTION_DIM, temperature=Config.TEMPERATURE):
        super(UnifiedEEGKeywordPredictor, self).__init__()

        # Multi-view EEG encoder
        self.eeg_encoder = UnifiedMultiViewEEGEncoder(
            in_channels, hidden_dim, projection_dim=projection_dim
        )

        # Label embedding - learn representations for each class
        self.label_embedding = nn.Parameter(torch.randn(num_labels, hidden_dim))
        nn.init.xavier_uniform_(self.label_embedding)

        # Similarity projection
        self.similarity_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temperature parameter for contrastive loss
        self.temperature = temperature

    def forward(self, x, x_augmented=None, views=None, views_aug=None):
        if x_augmented is not None and views is not None and views_aug is not None:
            # Training mode with contrastive learning and multi-view
            features, global_proj, view_projections, view_importance = self.eeg_encoder(
                x, views=views, return_all=True
            )

            # Get augmented features
            aug_features, aug_global_proj, aug_view_projections, _ = self.eeg_encoder(
                x_augmented, views=views_aug, return_all=True
            )

            # Project features for classification
            cls_features = self.similarity_proj(features)

            # Normalize embeddings for cosine similarity
            cls_features = F.normalize(cls_features, p=2, dim=1)
            label_emb_norm = F.normalize(self.label_embedding, p=2, dim=1)

            # Calculate similarity scores for classification
            similarity = torch.matmul(cls_features, label_emb_norm.t()) / self.temperature

            # Return all components for multi-objective training
            return similarity, global_proj, aug_global_proj, view_projections, aug_view_projections, view_importance

        else:
            # Inference mode
            features = self.eeg_encoder(x, views=views)

            # Project features for classification
            cls_features = self.similarity_proj(features)

            # Normalize embeddings for cosine similarity
            cls_features = F.normalize(cls_features, p=2, dim=1)
            label_emb_norm = F.normalize(self.label_embedding, p=2, dim=1)

            # Calculate similarity scores for classification
            similarity = torch.matmul(cls_features, label_emb_norm.t()) / self.temperature

            return similarity


class EnhancedMultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=Config.TEMPERATURE):
        super(EnhancedMultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.LARGE_NUM = 1e9  # Used for masking in logsumexp

    def forward(self, projections_1, projections_2, labels):
        """
        Compute Supervised Contrastive Loss.
        Args:
            projections_1: Tensor of shape [batch_size, projection_dim], e.g., global_proj from original.
            projections_2: Tensor of shape [batch_size, projection_dim], e.g., global_proj from augmented.
            labels: Tensor of shape [batch_size] with integer class labels.
        Returns:
            Scalar SupCon loss.
        """
        device = projections_1.device

        # Normalize projections
        projections_1 = F.normalize(projections_1, p=2, dim=1)
        projections_2 = F.normalize(projections_2, p=2, dim=1)

        # Concatenate projections to treat each original and its augmentation as separate items
        # for the purpose of finding positives within the same class.
        features = torch.cat([projections_1, projections_2], dim=0)  # [2*batch_size, projection_dim]
        batch_size_supcon = features.size(0)

        # Duplicate labels for the concatenated features
        supcon_labels = labels.repeat(2)  # [2*batch_size]

        # Create a mask for positive pairs (samples with the same label)
        # label_matrix[i,j] = 1 if supcon_labels[i] == supcon_labels[j]
        label_matrix = supcon_labels.contiguous().view(-1, 1)
        # Produces a matrix where mask[i,j] is 1 if item i and item j have the same label.
        mask = torch.eq(label_matrix, label_matrix.T).float().to(device)

        # Compute similarity matrix (cosine similarity as features are normalized)
        # sim_matrix[i,j] = cos_sim(features[i], features[j])
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [2*batch_size, 2*batch_size]

        # Mask out self-similarity from positives (diagonal elements)
        # We don't want log(exp(z_i . z_i / T) / ...) but log(exp(z_i . z_j / T) / ...) where i != j for positives.
        # However, for the denominator, all k != i are included.
        # The standard SupCon formulation includes z_i's own augmentation as a positive.
        # Let's refine the mask:
        # - Numerator: sum over positives p in P(i) where P(i) includes other samples of same class AND i's own augmentation.
        # - Denominator: sum over all k != i.

        # Mask for excluding self-contrast (i.e., features[i] vs features[i])
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(batch_size_supcon).view(-1, 1).to(device), 0)

        mask_positives = mask * logits_mask  # Positive pairs, excluding self

        # For numerical stability: subtract max logit
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()

        # Compute log_prob
        # Numerator: Sum over exp(sim(z_i, z_p)/T) for p in P(i) (positives for i)
        # Denominator: Sum over exp(sim(z_i, z_k)/T) for all k != i

        exp_logits = torch.exp(logits) * logits_mask  # Zero out diagonal for denominator sum
        log_denominator = torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)  # Add epsilon for stability

        # log_prob = sim_matrix - log_sum_exp over negatives (more complex)
        # Simpler: log_prob = specific_positive_sim - log_denominator

        # SupCon Loss: -log( sum_{p in P(i)} exp(z_i . z_p / T) / sum_{k != i} exp(z_i . z_k / T) )
        # This can be computed per positive pair and averaged.
        # Sum over all positives p for anchor i, then average over i.

        # (mask_positives * logits) gives dot products for positive pairs.
        # We need to sum these up for each anchor i.
        log_prob_positive_pairs = (mask_positives * logits) - log_denominator

        # The loss is - E_i [ (1/|P(i)|) * sum_{p in P(i)} log_prob_for_pair(i,p) ]
        # where log_prob_for_pair(i,p) = (z_i . z_p / T) - log(sum_{k!=i} exp(z_i . z_k / T))

        # Number of positive pairs for each anchor (excluding self)
        num_positives_per_anchor = mask_positives.sum(1)

        # Avoid division by zero if an anchor has no other positives in the batch
        # (e.g. if a class appears only once, its two augmentations are positives to each other)
        num_positives_per_anchor = torch.where(num_positives_per_anchor == 0, torch.ones_like(num_positives_per_anchor),
                                               num_positives_per_anchor)

        mean_log_prob_over_positives = (log_prob_positive_pairs).sum(1) / num_positives_per_anchor

        loss = -mean_log_prob_over_positives.mean()

        # The original SupCon paper uses a slightly different formulation for the average,
        # let's use a common PyTorch implementation style for clarity:
        # (from https://github.com/HobbitLong/SupContrast/blob/master/losses.py)

        # mask out self-contrast for positive pairs
        diag_mask = torch.eye(batch_size_supcon, device=device, dtype=torch.bool)
        mask_positives_no_self = mask.masked_fill(diag_mask, 0)  # All positives (label match), excluding self.

        # For the NCE loss, we want exp(sim(anchor, positive)) / sum_others exp(sim(anchor, other))
        # `sim_matrix` already has sim(anchor, other) / T
        # `logits` is sim_matrix - max_logit for stability

        # For each anchor, sum exp(sim(anchor, positive)) for all its positives
        exp_logits_stable = torch.exp(logits)  # Numerically stable exp_logits

        # Denominator sum: sum over all k (exp(sim(i,k)/T)) where k != i
        sum_exp_logits_denominator = (exp_logits_stable * logits_mask).sum(1, keepdim=True)  # Sum over k, for each i

        # log( exp(sim(i,p)/T) / sum_{k!=i} exp(sim(i,k)/T) )
        # = sim(i,p)/T - log(sum_{k!=i} exp(sim(i,k)/T))
        log_probs = logits - torch.log(sum_exp_logits_denominator + 1e-9)  # Add epsilon

        # Average log_probs over all positive pairs (p in P(i)) for each anchor i
        # Sum_{i} (1/|P(i)|) Sum_{p in P(i)} log_probs(i,p)

        # Number of true positives for each anchor (excluding self, but including its own other aug if same class)
        # mask_positives_no_self.sum(1) is |P(i)|

        # Check for anchors with no positives (should not happen if B > 1 for a class)
        num_positives = mask_positives_no_self.sum(1)

        # SupCon loss calculation
        # For each anchor `i`, sum `log_probs[i, p]` over all its positive counterparts `p`.
        # Then average this sum by `num_positives[i]`.
        # Finally, average over all anchors `i`.

        # (log_probs * mask_positives_no_self) effectively selects log_probs for positive pairs
        # sum(1) sums these selected log_probs for each anchor
        sum_log_probs_for_positives = (log_probs * mask_positives_no_self).sum(1)

        # Handle cases where num_positives is 0 to avoid NaN
        # If num_positives is 0 for an anchor, its contribution to loss is 0.
        loss_per_anchor = sum_log_probs_for_positives / (num_positives + 1e-9)  # add epsilon

        # Average over all anchors.
        # Note: SupCon paper averages over anchors that have at least one positive.
        # Here, we average over all 2*B samples. If num_positives is 0, loss_per_anchor is 0.
        loss = -loss_per_anchor.mean()

        return loss
