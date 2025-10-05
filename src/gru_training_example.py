from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.gru.modeling_gru import GRUForCausalLM
from fla.models.gru.configuration_gru import GRUConfig

from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

def train(
    model: GRUForCausalLM,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    max_epochs: int,
):
    model.train()
    for epoch in range(max_epochs):
        for inputs, targets in tqdm(train_dl, desc=f"Train Epoch {epoch}/{max_epochs}"):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            output = model(inputs, attention_mask=torch.ones_like(inputs))
            logits = output.logits
            loss = non_shifting_loss(logits, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

def eval(model: GRUForCausalLM, test_dl: DataLoader):
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
    data_cfg = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=64,
        vocab_size=65,
        batch_size=256,
        num_kv_pairs=16,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=37,
    )
    model_cfg = GRUConfig(
        # core architecture
        vocab_size=65,
        hidden_size=256,
        num_hidden_layers=2,
        dropout=0.0,
        bidirectional=False,

        # MLP (SwiGLU) - GRU also supports these
        hidden_ratio=4,
        intermediate_size=1024,
        hidden_act="swish",

        # norms and numerics
        norm_eps=1e-6,
        elementwise_affine=True,
        fuse_norm=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,

        # runtime semantics
        use_cache=False,
        tie_word_embeddings=False,
    )

    train_dl, test_dl = get_dataloaders(data_cfg)

    # model_cfg = gru_cfg
    # we will use float32 for ours
    model = GRUForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.float32)  
    torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), 
                            lr=1e-4, 
                            weight_decay=1e-6)

    # from ipdb import set_trace; set_trace()
    model = train(model, train_dl, optimizer, 30)
    accuracy = eval(model, test_dl)
    print(f"Accuracy: {accuracy}")
    # from ipdb import set_trace; set_trace()
