import appdirs
from importlib.resources import files
from pathlib import Path
import yaml
import collections.abc
from collections import UserDict

def attrlist_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode):
    return [v.upper() if isinstance(v, str) else v for v in loader.construct_sequence(node)]

def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!attrs", attrlist_constructor)
    return loader

def _update(target, source):
    for k, v in source.items():
        if isinstance(v, collections.abc.Mapping):
            target[k] = _update(target.get(k, {}), v)
        else:
            target[k] = v
    return target

def get_user_config_path():
    return Path(appdirs.user_config_dir(appname="ori"))

def get_default_config():
    with open(files('ori').joinpath('config.yaml'), 'rb') as _config:
        return yaml.load(_config, Loader=get_loader())

def get_config():
    user_config_path = get_user_config_path()
    if not user_config_path.exists():
        user_config_path.mkdir(parents=True, exist_ok=True)
    user_config_file = user_config_path/'config.yaml'
    default_config = get_default_config()
    user_config = {}
    if user_config_file.exists():
        with open(user_config_file, 'rb') as _user_config:
            user_config = yaml.load(_user_config, Loader = get_loader())
    return _update(user_config, default_config)
        