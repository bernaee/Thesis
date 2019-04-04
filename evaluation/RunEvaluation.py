import os

corpus_path = '/home/berna/PycharmProjects/MS/Deep-BGT/data/corpora/sharedtask-data-master/1.1'
output_path = '/home/berna/PycharmProjects/MS/Deep-BGT/results'
model_name = '02'
languages = os.listdir(os.path.join(output_path, model_name))
languages=['TR']
commands = """#!/bin/sh \n alias python='python3' \n export PATH="$PATH:/home/berna/PycharmProjects/MS/ParsemeEvaluation" \n """
for lang in languages:
    input_path = os.path.join(corpus_path, lang)
    for exp_no in ['1', '2', '3']:
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
with open('/home/berna/PycharmProjects/MS/ParsemeEvaluation/eval.sh', 'w') as the_file:
    the_file.write(commands)
