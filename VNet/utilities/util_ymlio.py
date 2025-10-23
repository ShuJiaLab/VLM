from ruamel.yaml import YAML

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        yaml=YAML(typ='rt')
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def write_yaml(yaml_file, data):
    with open(yaml_file, 'w') as stream:
        yaml=YAML(typ='rt')
        try:
            yaml.dump(data, stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    from os import system, name
    from pprint import pprint

    system('cls' if name == 'nt' else 'clear')

    params = read_yaml('L:/HRFLFMnet/dl_pytorch_100x_v2/paraymls/DefaultParams_test.yml')
    pprint(params,sort_dicts=False)

    write_yaml('L:/HRFLFMnet/dl_pytorch_100x_v2/paraymls/DefaultParams.yml', params)