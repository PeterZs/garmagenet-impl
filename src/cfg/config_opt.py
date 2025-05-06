import yaml
from easydict import EasyDict as edict

def load_yaml(file_path):
    try:
        with open(file_path, "r") as f:
            return edict(yaml.safe_load(f) or {})  # 读入后转换为 EasyDict
    except FileNotFoundError:
        return edict()


def merge_configs(default_config, custom_config):
    """
    递归合并两个 EasyDict, custom_config会覆盖default_config的键值相同部分
    Args:
        default_config:
        custom_config:
    Returns:
    """
    for key, value in custom_config.items():
        default_config[key] = (
            merge_configs(default_config[key], value)
                if isinstance(value, dict) and isinstance(default_config.get(key), dict)
            else value
        )
    return default_config

def get_GarmageNet_cfg(Base_config, yaml_file = None):
    if not isinstance(yaml_file, list):
        yaml_file = [yaml_file]

    for fp in yaml_file:
        f_yaml = load_yaml(fp)
        merge_configs(Base_config, f_yaml)

    return Base_config