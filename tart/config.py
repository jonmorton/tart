from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf


@dataclass
class DataConfig:
    num_workers: int = 4
    root: str = MISSING


@dataclass
class OptimConfig:
    lr: float = 2e-4
    warmup_steps: int = 1000
    total_steps: int = 50000

    weight_decay: float = 0.05
    grad_norm_clip: float = 1.0
    logit_reg_weight: float = 1e-4

    batch_size: int = 64
    grad_accum_steps: int = 1

    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8


@dataclass
class ModelConfig:
    layout: str = "gpt2"
    seq_len: int = 512
    num_layers: int = 12
    embed_dim: int = 512


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    out_dir: str = "./_train_out"
    checkpoint_period: int = 2000
    log_period: int = 10
    log_hists_period: int = 300
    val_period: int = 300
    val_examples: int = 8192
    random_seed: int = 1337
    compile: bool = True


cfg: Config = OmegaConf.structured(Config)


class cfg_util:
    @staticmethod
    def merge_yaml(path):
        return cfg.merge_with(OmegaConf.load(path))

    @staticmethod
    def merge_dotlist(args):
        return cfg.merge_with_dotlist(args)

    @staticmethod
    def save_yaml(path):
        return OmegaConf.save(cfg, path)

    @staticmethod
    def to_yaml():
        return OmegaConf.to_yaml(cfg)

    @staticmethod
    def to_dict():
        return OmegaConf.to_container(cfg)
