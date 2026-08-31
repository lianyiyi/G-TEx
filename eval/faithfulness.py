import argparse
import json
import os
from collections import defaultdict


def list_txt_files(startpath='.'):
    paths = []
    for root, dirs, files in os.walk(startpath):
        for name in dirs + files:
            full_path = os.path.join(root, name)
            if 'txt' in full_path:
                paths.append(full_path)
    return paths


def make_dict(files, target_id):
    """Group inputs/hypotheses by reference text for a given run id.

    When the files come from the counterfactual (faithful) run, several
    perturbed generations share the same reference, so values are collected in a
    list; otherwise each reference maps to a single generation.
    """
    inputs = hyps = refs = None
    for x in files:
        if 'inputs_0.0003' + target_id in x:
            inputs = open(x, 'r').readlines()
        elif 'hypos_0.0003' + target_id in x:
            hyps = open(x, 'r').readlines()
        elif 'refs_0.0003' + target_id in x:
            refs = open(x, 'r').readlines()

    generation = defaultdict(list)
    is_faithful_run = 'faithful' in files[0]
    for a, b, c in zip(inputs, hyps, refs):
        a, b, c = a.strip(), b.strip(), c.strip()
        new_dict = {'hyp': b, 'input': a}
        if is_faithful_run:
            generation[c].append(new_dict)
        else:
            generation[c] = new_dict
    return generation


def calculate_faithfulness(baseline_path, inserted_word_path):
    with open(baseline_path, 'r') as b:
        base = json.load(b)
    with open(inserted_word_path, 'r') as i:
        change = json.load(i)

    # Attach each baseline example to its counterfactual generations, dropping
    # any baseline example without a matching perturbed run.
    for key, values in change.items():
        if key in base:
            base[key]['inserted_token'] = values
    for key in list(base.keys()):
        if 'inserted_token' not in base[key]:
            del base[key]

    a = 0
    b = 0
    n = len(base)

    for key, value in base.items():
        pre_label = value['hyp'].split('explanation:')[0]

        after_labels = [v['hyp'].split('explanation:')[0] for v in value['inserted_token']]
        for l in after_labels:
            if l.lower() not in pre_label.lower():
                a += 1
                break

        inserted_tokens = [v['input'].split(' ')[-1] for v in value['inserted_token']]
        generated_text = [v['hyp'].split('explanation:')[-1] for v in value['inserted_token']]

        for l, t, h in list(zip(after_labels, inserted_tokens, generated_text)):
            if l.lower() not in pre_label.lower():
                if t.lower() not in h.lower():
                    b += 1
                    break

    counter = a / n if n > 0 else 0
    counter_unfaith = b / a if a > 0 else 0
    total_unfaith = b / n if n > 0 else 0
    return counter, counter_unfaith, total_unfaith


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', required=True,
                        help='Directory holding the baseline generation txt files.')
    parser.add_argument('--counterfactual_dir', required=True,
                        help='Directory holding the counterfactual (word-inserted) generation txt files.')
    parser.add_argument('--target_ids', nargs='+', required=True,
                        help='Run ids (test hashes) to evaluate.')
    args = parser.parse_args()

    baseline_files = list_txt_files(args.baseline_dir)
    counterfactual_files = list_txt_files(args.counterfactual_dir)

    for target_id in args.target_ids:
        base_dict = make_dict(baseline_files, target_id)
        change_dict = make_dict(counterfactual_files, target_id)

        baseline_json = os.path.join(args.baseline_dir, 'all_text.json')
        counterfactual_json = os.path.join(args.counterfactual_dir, 'all_text.json')
        with open(baseline_json, 'w') as f:
            json.dump(base_dict, f)
        with open(counterfactual_json, 'w') as f:
            json.dump(change_dict, f)

        counter, counter_unfaith, total_unfaith = calculate_faithfulness(baseline_json, counterfactual_json)
        print(f'{target_id}\tcounter={counter:.4f}\tcounter_unfaith={counter_unfaith:.4f}\ttotal_unfaith={total_unfaith:.4f}')


if __name__ == '__main__':
    main()
