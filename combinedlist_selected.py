import json
import random

def proportional_random_sample(data_dict, n, verbose=True):

    counts = {k: len(v) for k, v in data_dict.items()}
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return []

    picks_info = []
    sum_picks = 0
    for k, cnt in counts.items():
        ratio = cnt / total
        exact_float = ratio * n
        picked = int(round(exact_float))
        picks_info.append({
            'key': k,
            'count': cnt,
            'exact_float': exact_float,
            'picked': picked
        })
        sum_picks += picked

    diff = sum_picks - n

    if diff != 0:
        
        for item in picks_info:
            item['fractional_part'] = item['exact_float'] - item['picked']

        if diff > 0:
            picks_info.sort(key=lambda x: x['fractional_part'])
            reduce_count = diff
            i = 0
            while reduce_count > 0 and i < len(picks_info):
                if picks_info[i]['picked'] > 0:
                    picks_info[i]['picked'] -= 1
                    reduce_count -= 1
                i += 1
        else:
            picks_info.sort(key=lambda x: x['fractional_part'], reverse=True)
            add_count = -diff
            i = 0
            while add_count > 0 and i < len(picks_info):
                if picks_info[i]['picked'] < picks_info[i]['count']:
                    picks_info[i]['picked'] += 1
                    add_count -= 1
                i += 1

    result = []
    for item in picks_info:
        k = item['key']
        want = min(item['picked'], len(data_dict[k]))

        picked_items = random.sample(data_dict[k], want)
        result.extend(picked_items)
        if verbose:
            print(f"Key={k}: It has {len(data_dict[k])} elements, aim to select {want} elements, actually select {len(picked_items)} elements")

    return result


if __name__ == "__main__":

    json_file_path = "stored_data/combined_dict.json"
    n = 100

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_list = proportional_random_sample(data, n)

    out_file_path = "stored_data/new_combined_list_100.txt"
    with open(out_file_path, 'w', encoding='utf-8') as f:
        for elem in new_list:
            f.write(str(elem) + "\n")

    print(f"In total select {len(new_list)} elements, written in {out_file_path}")
