from parlai.core.params import ParlaiParser
import os
from parlai.scripts.build_dict import build_dict
from parlai.tasks.convai2.build import build


def setup_args(data_file):
    parser = ParlaiParser(add_model_args=True, add_parlai_args=True)
    parser.set_defaults(task='convai2:self_original',
                        dict_tokenizer='split',
                        data_file=data_file,
                        dict_maxexs=-1)
    return parser


def read_personae_from_file(path):
    """
    return [[persona_1, persona_2, persona_3], ...]
    """
    try:
        with open(path, encoding='utf-8') as f:
            personae = []
            mode = 0
            last_mode = None
            this_persona = []
            for line in f:
                line = line[:-1]
                if 'your persona:' in line:
                    mode = 1
                elif 'partner\'s persona:' in line:
                    mode = 2
                    line = line.replace('partner\'s persona:', 'your persona:')
                else:
                    last_mode = None
                    continue

                if mode != last_mode:
                    if this_persona:
                        personae.append(tuple(this_persona))
                        this_persona = []
                line = line.split(': ', maxsplit=2)[1]
                this_persona.append(line)
                last_mode = mode

            if this_persona:
                personae.append(tuple(this_persona))
    except UnicodeDecodeError as e:
        print("    {} is not in utf-8 format".format(path))
        print("    ", e)
        personae = []
    return personae


def convert(personae, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        output = ''
        index = 1
        for persona in personae:
            desc = ' '.join(persona)
            output += '{} {}\n'.format(index, desc)
        f.write(output)
    return output


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    return os.path.join('data', 'ConvAI2')


def _build_dict(opt):
    build_dict(opt)


def _build_personae_data(opt):
    dir_path = _path(opt)
    datatype = "train" if "train" in opt["data_file"] else "valid"
    personae = read_personae_from_file(opt["data_file"])
    if 'original' in opt["data_file"]:
        convert(personae, os.path.join(dir_path, datatype + '_receiver_both_original.txt'))
    else:
        convert(personae, os.path.join(dir_path, datatype + '_receiver_both_revised.txt'))


def build_data(data_file):
    parser = setup_args(data_file)
    opt = parser.parse_args([])
    _build_dict(opt)
    _build_personae_data(opt)


if __name__ == '__main__':
    build_data("data\\ConvAI2\\train_both_original.txt")
    build_data("data\\ConvAI2\\valid_both_original.txt")
    build_data("data\\ConvAI2\\train_both_revised.txt")
    build_data("data\\ConvAI2\\valid_both_revised.txt")


