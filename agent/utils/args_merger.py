import yaml 
import argparse



def merge(args):
    """
    Merges configuration data from a YAML file and command-line arguments.

    Args:
        args (argparse.Namespace): An object containing command-line arguments, where
            `args.config` specifies the path to the YAML configuration file.

    Returns:
        argparse.Namespace: An updated Namespace object containing the merged configuration,
        with keys from both the YAML file and the command-line arguments.
    """
    with open(args.config, "r") as file:
       yaml_data = yaml.safe_load(file)
    yaml_dict = flatten_dict(yaml_data)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    combined_config = {**yaml_dict, **args_dict}
    new_args = argparse.Namespace(**combined_config)
    return new_args




def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary by combining keys into a single level using a specified separator.

    Args:
        d (dict): The dictionary to flatten. Can contain nested dictionaries as values.
        parent_key (str, optional): The base key to prepend to each key in the flattened dictionary. Defaults to an empty string.
        sep (str, optional): The separator to use when combining keys. Defaults to '.'.

    Returns:
        dict: A flattened dictionary where nested keys are combined into a single key separated by `sep`.

    Example:
        >>> nested_dict = {'a': {'b': {'c': 1}}, 'd': 2}
        >>> flatten_dict(nested_dict)
        {'a.b.c': 1, 'd': 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

