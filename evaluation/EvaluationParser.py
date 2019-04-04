import pandas as pd
import os


def parse_results(results, metric):
    sub_results = results.split('\n')
    info = dict()
    sub_results = sub_results[1:]
    sub_results = list(filter(lambda e: e != '', sub_results))
    for sb in sub_results:
        sb_list = sb.split(' ')
        if metric == 'global':
            if not info.get(metric):
                info[metric] = dict()
            type = sb_list[1][:-1]
            prec = round(float(sb_list[2].split('=')[2]) * 100, 2)
            recall = round(float(sb_list[3].split('=')[2]) * 100, 2)
            f_measure = round(float(sb_list[4].split('=')[1]) * 100, 2)
            info[metric][type + '-Prec'] = prec
            info[metric][type + '-Recall'] = recall
            info[metric][type + '-F-measure'] = f_measure
        else:
            metric_type = sb_list[1][:-1]
            if not info.get(metric_type):
                info[metric_type] = dict()
            type = sb_list[2][:-1]
            # print(type)
            # print(metric_type)
            if type == 'MWE-proportion':
                gold = sb_list[3].split('=')[1].split('/')
                pred = sb_list[4].split('=')[1].split('/')
                info[metric_type][type + '-Gold'] = int(gold[0])
                info[metric_type][type + '-Pred'] = int(pred[0])
            if type == 'MWE-based' or type == 'Tok-based':
                prec = round(float(sb_list[3].split('=')[2]) * 100, 2)
                recall = round(float(sb_list[4].split('=')[2]) * 100, 2)
                f_measure = round(float(sb_list[5].split('=')[1]) * 100, 2)
                info[metric_type][type + '-Prec'] = prec
                info[metric_type][type + '-Recall'] = recall
                info[metric_type][type + '-F-measure'] = f_measure

    df = pd.DataFrame(info).T
    df['Metric'] = metric
    return df


output_path = '/home/berna/PycharmProjects/MS/Deep-BGT/results'
model_name = '03'
languages = os.listdir(os.path.join(output_path, model_name))
# languages=['','','']        model_name

df_all = pd.DataFrame()
df_global_all = pd.DataFrame()
for lang in languages:
    for exp_no in ['1', '2', '3']:
        eval_path = os.path.join(output_path, model_name, lang, exp_no, 'eval.txt')
        with open(eval_path, encoding='utf-8') as f:
            corpus = f.read()
        categories = corpus.split('##')
        df_global = parse_results(categories[1], 'global')
        per_category = parse_results(categories[2], 'category')
        continuity = parse_results(categories[3], 'continuity')
        multi_token = parse_results(categories[4], 'multi-token')
        seen_in_train = parse_results(categories[5], 'seen-in-train')
        iden_to_train = parse_results(categories[6], 'identical-to-train')
        df = pd.concat([per_category, continuity, multi_token, seen_in_train, iden_to_train])
        df['Lang'] = lang
        df['Model'] = model_name
        df['ExpNo'] = exp_no
        df_all = df_all.append(df)
        df_global['Lang'] = lang
        df_global['Model'] = model_name
        df_global['ExpNo'] = exp_no
        df_global_all = df_global_all.append(df_global)

df_all.reset_index(inplace=True)
df_all = df_all.rename(columns={'index': 'SubMetric'})
df_all.to_csv(os.path.join(output_path, '%s_res.csv' % model_name), index=False)
df_global_all.to_csv(os.path.join(output_path, '%s_global_res.csv' % model_name), index=False)
