import json
import os


def build_data(self_non_original, save_path):
    with open(self_non_original, "r", encoding="utf8") as f:
        dialogues = []
        your_persona = None
        other_persona = None
        start_sens = None
        for line in f.readlines():
            # remove number and space
            sep = line.find(' ')
            number = line[:sep]
            line = line[sep + 1:-1]
            if number == '1':
                your_persona = []
                other_persona = []
                start_sens = None
            # not persona
            if '\t' in line and start_sens is None:
                # first sentence
                other_text, your_text = line.split('\t')
                if other_text == '__SILENCE__':
                    other_text, your_text = your_text, other_text
                    other_persona, your_persona = your_persona, other_persona
                # place holder
                start_sens = other_text
                other_persona.append(other_text)
                # add into dialog
                temp_text = ''
                for ind, text in enumerate(other_persona):
                    temp_text += '{} {}\n'.format(ind + 1, text)
                dialogues.append(temp_text)
            elif '\t' in line:
                # second sentence
                continue
            else:
                if 'your' in line:
                    your_persona.append(line.strip())
                else:
                    other_persona.append(line.strip())
        save_file = open(save_path, "w", encoding="utf8")
        for text in dialogues:
            save_file.write(text)


if __name__ == '__main__':
    build_data("..\\..\\data\\ConvAI2\\train_both_original_no_cands.txt",
               "..\\..\\data\\ConvAI2\\train_self_original_selfplay.txt")
    build_data("..\\..\\data\\ConvAI2\\valid_both_original_no_cands.txt",
               "..\\..\\data\\ConvAI2\\valid_self_original_selfplay.txt")
    build_data("..\\..\\data\\ConvAI2\\train_both_revised_no_cands.txt",
               "..\\..\\data\\ConvAI2\\train_self_revised_selfplay.txt")
    build_data("..\\..\\data\\ConvAI2\\valid_both_revised_no_cands.txt",
               "..\\..\\data\\ConvAI2\\valid_self_revised_selfplay.txt")
