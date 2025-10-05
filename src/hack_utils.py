import torch 
import torch.nn.functional as F

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor):
    '''
    Compute the accuracy of the model's predictions.
    Args:
        logits: The logits of the model's predictions. Shape: (batch_size, seq_len, vocab_size).
        targets: The targets of the model's predictions. Shape: (batch_size, seq_len).
    Returns:
        The accuracy of the model's predictions.
    '''
    predicted_labels = torch.argmax(logits, dim=-1, keepdim=False)
    valid_mask = targets != -100
    correct = (predicted_labels == targets) & valid_mask
    total = valid_mask.sum()
    return correct.sum() / total

def non_shifting_loss(logits: torch.Tensor, targets: torch.Tensor):
    '''
    Compute the non-shifting loss of the model's predictions.
    Args:
        logits: The logits of the model's predictions. Shape: (batch_size, seq_len, vocab_size).
        targets: The targets of the model's predictions. Shape: (batch_size, seq_len).
    Returns:
        The non-shifting loss of the model's predictions.
    '''
    targets = targets.to(logits.device)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))