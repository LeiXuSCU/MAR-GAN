import json
import os

import ruamel.yaml
from options.options import Options


def load_config_yml(resource_path):
    resource_path = open(resource_path, 'r', encoding='utf-8')
    resource = resource_path.read()
    data = ruamel.yaml.load(resource, Loader=ruamel.yaml.Loader)
    return data


def load_obj_config(filename='application.yml'):
    project_root = os.path.dirname(os.path.dirname(__file__))
    config = load_config_yml(os.path.join(project_root, 'resource', filename))
    opts = Options(config)
    setattr(opts, 'project_root', project_root)
    return opts


def initial_config(filename='application.yml', is_train=True):
    opts = load_obj_config(filename)
    setattr(opts, 'is_train', is_train)

    input_path: str
    if is_train:
        input_path = os.path.join(opts.project_root, opts.train_path)
        if not os.path.exists(input_path):
            raise Exception('Oops... Train data {} nonexistent'.format(input_path))
        if opts.pretrained.enable:
            pretrained_model = os.path.join(opts.project_root, opts.pretrained.path)
            if not os.path.exists(pretrained_model):
                raise Exception('Oops... pretrained data {} nonexistent'.format(pretrained_model))
            opts.pretrained.path = pretrained_model
    else:
        input_path = os.path.join(opts.project_root, opts.test_path)
        if not os.path.exists(input_path):
            raise Exception('Oops... Test data {} nonexistent'.format(input_path))
        checkpoint_dir = os.path.join(opts.project_root, opts.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            raise Exception('Oops... checkpoint data {} nonexistent'.format(checkpoint_dir))

    setattr(opts, 'input_path', input_path)

    target_dir = opts.checkpoint_path if is_train else opts.result_path
    target_dir = os.path.join(opts.project_root, target_dir)

    if os.path.exists(target_dir):
        remove_dir(target_dir)
    os.makedirs(target_dir)

    print('------------- Configuration Info {} -------------'.format('Train' if is_train else 'Test'))
    print(json.dumps(opts, default=lambda o: o.__dict__, indent=4, sort_keys=True))
    with open(os.path.join(target_dir, 'opt.json'), 'w') as f:
        json.dump(opts, f, default=lambda o: o.__dict__, indent=4, sort_keys=True)
    return opts


def remove_all_in_dir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            remove_all_in_dir(c_path)
        else:
            os.remove(c_path)


def remove_dir(path):
    remove_all_in_dir(path)
    os.removedirs(path)
