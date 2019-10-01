import os
import argparse

parser = argparse.ArgumentParser(prog='deep-bgt')
parser.add_argument('-corpus', '--corpus_path')
parser.add_argument('-output', '--output_path')
parser.add_argument('-parseme', '--parseme_eval_path')
parser.add_argument('-model', '--model_name')
parser.add_argument('-exp', '--exp-no')

args = parser.parse_args()
corpus_path = args.corpus_path  # '/home/berna/PycharmProjects/Thesis/data/corpora/sharedtask-data-master/1.1'
output_path = args.output_path  # '/home/berna/PycharmProjects/Thesis/trials'
parseme_eval_path = args.parseme_eval_path  # '/home/berna/PycharmProjects/ParsemeEvaluation'
model_name = args.model_name  # '11'
exp_numbers = args.exp_no  # '1', '2', '3'
exp_numbers = exp_numbers.split(',')
languages = os.listdir(os.path.join(output_path, model_name))

commands = """#!/bin/sh \n alias python='python3' \n export PATH="$PATH:%s" \n """ % parseme_eval_path
for lang in languages:
    input_path = os.path.join(corpus_path, lang)
    for exp_no in exp_numbers:
        msg = """ echo "Evaluating %s-%s..." """ % (lang, exp_no)
        commands = commands + msg + '\n'
        model_path = os.path.join(output_path, model_name)
        train_output_path = os.path.join(model_path, lang)
        test_output_path = os.path.join(train_output_path, exp_no)

        train_path = os.path.join(input_path, 'train.cupt')
        test_gold_path = os.path.join(input_path, 'test.cupt')
        test_path = os.path.join(test_output_path, 'test_tagged.cupt')
        eval_path = os.path.join(test_output_path, 'eval.txt')
        command = 'python evaluate.py --gold %s --pred %s --train %s > %s' % (
            test_gold_path, test_path, train_path, eval_path)
        commands = commands + command + '\n'
#
eval_cmd = os.path.join(output_path, 'eval.cmd')
with open(os.path.join(parseme_eval_path, 'eval.sh', 'w')) as the_file:
    the_file.write(commands)
