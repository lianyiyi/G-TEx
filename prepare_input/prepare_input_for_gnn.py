import argparse
import json

from tqdm import tqdm


def build_high_tokens(record):
    token_with_score = record['highlights_for_all_sentences']
    token_with_score = sorted(token_with_score, key=lambda x: x['score'], reverse=True)
    # +10 offset accounts for the prompt tokens prepended to every source input.
    return [t['sub_token_pos'] + 10 for t in token_with_score if t['text'] != '</s>']


def build_token_pairs(record):
    token_with_score = record['token_pair_explanations']
    token_with_score = sorted(token_with_score, key=lambda x: x['score'], reverse=True)

    token_pairs = []
    for t in token_with_score:
        start_token = list(range(t['start_subtoken_pos_1'], t['end_subtoken_pos_1'] + 1))
        end_token = list(range(t['start_subtoken_pos_2'], t['end_subtoken_pos_2'] + 1))
        token_pairs.append([start_token, end_token])
    return token_pairs


def convert(input_path, mode):
    cleaned_samples = []
    for line in tqdm(open(input_path)):
        record = json.loads(line)
        sample = {
            'text1': record['part1'],
            'text2': record['part2'],
            'label': str(record['label_index']),
            'exp1': record['gold_explanation_1'],
            'exp2': record['gold_explanation_2'],
        }
        if mode == 'high_tokens':
            # high_tokens concatenates the three gold explanations as the target.
            sample['exp3'] = (
                f"{record['gold_explanation_1']} {record['gold_explanation_2']} {record['gold_explanation_3']}"
            )
            sample['tokens'] = build_high_tokens(record)
        else:
            sample['exp3'] = record['gold_explanation_3']
            sample['tokens'] = build_token_pairs(record)
        cleaned_samples.append(sample)
    return cleaned_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Raw JSON-lines annotation file.')
    parser.add_argument('--output', required=True, help='Destination JSON file.')
    parser.add_argument('--mode', required=True, choices=['high_tokens', 'span_pair', 'token_pair'],
                        help='Highlight graph construction mode.')
    args = parser.parse_args()

    samples = convert(args.input, args.mode)
    with open(args.output, 'w') as f:
        json.dump(samples, f)


if __name__ == '__main__':
    main()
