import os
import yaml
from typing import Dict, List

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


## register the tag handler
yaml.add_constructor('!join', join)


def _update_dict(d: dict, params: dict):
    print("Overwriting config parameters:")
    for k, v in params.items():
        *path, key = k.split(".")
        inner_dict = d
        for path_key in path:
            inner_dict = inner_dict[path_key]
        old_v = inner_dict.get(key)
        inner_dict[key] = v
        print(f"    ", f"{k} ".ljust(50, '.'), f"{old_v} -> {v}")
    return d


def save_config(config: dict, directory: str, name='config.yml'):
    # read config to a file
    os.makedirs(directory, exist_ok=True)
    fp = os.path.join(directory, name)
    with open(fp, 'w') as f:
        yaml.dump(config, f)


def read_config(cfg_path: str, **kwargs) -> dict:
    # read config to a dictionary
    with open(cfg_path) as cfg:
        cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

    return cfg_yaml