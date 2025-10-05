from fla.models.gru.configuration_gru import GRUConfig

cfg = GRUConfig(
    # core architecture
    vocab_size=769,
    hidden_size=256,
    num_hidden_layers=2,
    dropout=0.0,
    bidirectional=False,

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
