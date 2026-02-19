import torch

import torch.nn.functional as F

def infoNCE(query, positive, negatives, temperature=0.07):
    """
    InfoNCE loss implementation.
    
    Args:
        query: Query embeddings, shape (batch_size, embedding_dim)
        positive: Positive embeddings, shape (batch_size, embedding_dim)
        negatives: Negative embeddings, shape (batch_size, num_negatives, embedding_dim)
        temperature: Temperature parameter for scaling
    
    Returns:
        loss: InfoNCE loss value
    """
    # Normalize embeddings
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    # Compute logits
    pos_logits = torch.sum(query * positive, dim=-1, keepdim=True) / temperature # shape (batch_size, 1)
    # shape (batch_size, num_negatives) = (batch_size, num_negatives, embedding_dim) @ (batch_size, embedding_dim, 1) -> (batch_size, num_negatives, 1) -> (batch_size, num_negatives)
    neg_logits = torch.matmul(negatives, query.unsqueeze(-1)).squeeze(-1) / temperature # shape (batch_size, num_negatives)
    

    # Concatenate positive and negative logits

    # [[0.8],  [[0.1, 0.2],  
    # [0.9]]   [0.3, 0.1]] 
    
  
    logits = torch.cat([pos_logits, neg_logits], dim=1)

    # [[0.8,0.1,0.2],
    # [0.9,0.3,0.1]]

    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    # tensor([0,0]) shape (batch_size,)
    
    # Compute cross entropy loss
    # logits: shape (batch_size, class_count)
    # labels:  shape (batch_size,)
    loss = F.cross_entropy(logits, labels)
   
    
    return loss