import yaml

def load_config(config_path: str):
    """
    从 .yaml 文件加载配置。
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config