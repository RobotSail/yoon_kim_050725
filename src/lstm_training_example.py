from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '/workspace/home/lab/osilkin/yoon_kim_050725/src')

from models.lstm.modeling_lstm import LSTMForCausalLM
from models.lstm.configuration_lstm import LSTMConfig
from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy


import random
import numpy as np

MANUAL_SEED = 37
def set_seed(seed: int):
    """set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for full determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# try muon
from muon_fsdp2 import Muon

def muon_param_names(model: LSTMForCausalLM):
    valid_params = []
    for name, param in model.named_parameters():
        if 'embeddings' in name or 'lm_head' in name:
            continue
        if param.dim() == 1:
            continue
        valid_params.append(name)
    return valid_params

def create_muon_optimizer(model: LSTMForCausalLM, lr: float, adamw_lr: float):
    muon_names = muon_param_names(model)
    muon_optimizer = Muon(
        param_groups=[
            {
                "params": [p for n, p in model.named_parameters() if n in muon_names],
                "lr": lr,
                "use_muon": True,
                "rms_scale": True,
                "nesterov": True,
                "ns_steps": 5,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in muon_names],
                "lr": adamw_lr,
                "use_muon": False,
                "betas": (0.9, 0.95),
            }
        ]
    )
    return muon_optimizer




def train(
    model: LSTMForCausalLM,
    train_dl: DataLoader,
    test_dl: DataLoader,
    optimizer: optim.Optimizer,
    max_epochs: int,
    l1_strength: float = None,
):
    model.train()
    beta = 0.9
    for epoch in range(max_epochs):
        gradnorm_ema = 0.0
        for inputs, targets in tqdm(train_dl, desc=f"Train Epoch {epoch}/{max_epochs}"):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            output = model(inputs)
            logits = output.logits
            loss = non_shifting_loss(logits, targets)

            # add l1
            if l1_strength is not None:
                l1_penalty = l1_strength * sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_penalty

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradnorm_ema = beta * gradnorm_ema + (1 - beta) * grad_norm
            optimizer.step()

        accuracy = eval(model, test_dl)
        print(f"Accuracy: {accuracy}")
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}, grad norm: {grad_norm:.5f}")
        model.train()
        
    return model

def eval(model: LSTMForCausalLM, test_dl: DataLoader):
    model.eval()
    all_logits = []
    all_targets = []
    for inputs, targets in test_dl:
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        output = model(inputs)
        logits = output.logits
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return compute_accuracy(all_logits, all_targets)

if __name__ == "__main__":
    set_seed(MANUAL_SEED)

    kv = 64
    lr = 1e-4
    # muon_adamw_lr = 1e-3
    num_hidden_layers = 2
    hidden_size = 1024
    proj_size = 256
    l1_strength = 1e-7
    num_epochs = 60
        #     input_seq_len=4 * kv,
        # vocab_size=4 * kv + 1,
        # batch_size=256,
        # num_kv_pairs=kv,

    # Use same data config as transformer baseline
    data_cfg = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=4 * kv,
        vocab_size=4 * kv + 1,
        batch_size=256,
        num_kv_pairs=kv,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=MANUAL_SEED,
    )

    # LSTM configuration with similar capacity to transformer baseline
    model_cfg = LSTMConfig(
        # core architecture
        vocab_size=4 * kv + 1,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        proj_size=proj_size,
        dropout=0.1,

        residual_connection=True,
        residual_scale=1.0,
        use_residual_layernorm=True,

        # enable self-attention
        # attention (optional)
        use_self_attention=True,
        num_attention_heads = 8,
        attention_dropout = 0.1,
        use_attention_layernorm = True,
        attn_residual_scale = 1.0,


        # norms and numerics
        norm_eps=1e-6,
        elementwise_affine=True,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,

        # runtime semantics
        use_cache=False,
    )

    train_dl, test_dl = get_dataloaders(data_cfg)

    # Create model and move to GPU 4 (cuda:4)
    model = LSTMForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.float32)
    print(f"Number of parameters: {model.num_parameters():,}")

    model = torch.compile(model)
    model.train()


    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        # weight_decay=1e-6,
        betas=(0.9, 0.95),
    )

    # optimizer = create_muon_optimizer(model, lr, muon_adamw_lr)



    model = train(model, train_dl, test_dl, optimizer, num_epochs, l1_strength=l1_strength)
    accuracy = eval(model, test_dl)
    print(f"Accuracy: {accuracy}")
