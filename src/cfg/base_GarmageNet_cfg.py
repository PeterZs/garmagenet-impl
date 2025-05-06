from config_opt import *
from easydict import EasyDict as edict

GarmageNet_cfg = edict()
GarmageNet_cfg.VAE = edict()


if __name__ == '__main__':
    get_GarmageNet_cfg(GarmageNet_cfg.VAE, yaml_file = ["src/cfg/config/_for_test.yaml"])