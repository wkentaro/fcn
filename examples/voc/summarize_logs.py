#!/usr/bin/env python

import argparse
import datetime
import json
import os
import os.path as osp
import yaml

import dateutil.parser
import pandas
import tabulate


def summarize_log(logs_dir, name, params_keys, target_key, objective, show_active):
    try:
        params = yaml.load(open(osp.join(logs_dir, name, 'params.yaml')))
    except Exception:
        return None, None, osp.join(logs_dir, name)

    log_file = osp.join(logs_dir, name, 'log')
    try:
        df = pandas.DataFrame(json.load(open(log_file)))
    except Exception:
        return None, None, osp.join(logs_dir, name)

    try:
        if objective == 'min':
            idx = df[target_key].idxmin()
        else:
            idx = df[target_key].idxmax()
    except Exception:
        idx = None

    dfi = df.ix[idx] if idx else None
    row = []
    is_active = True
    for key in keys:
        if key == 'name':
            row.append(name)
        elif key in ['epoch', 'iteration']:
            if dfi is None:
                value = '<none>'
            else:
                value = '%d' % dfi[key]
            max_value = df[key].max()
            row.append('%s /%d' % (value, max_value))
        elif key == 'last_time':
            if df is None:
                value = '<none>'
            else:
                elapsed_time = df['elapsed_time'].max()
                value = params.get('timestamp', '<none>')
                if value is not '<none>':
                    value = dateutil.parser.parse(value)
                    value += datetime.timedelta(seconds=elapsed_time)
                    now = datetime.datetime.now()
                    value = now - value
                    if value > datetime.timedelta(minutes=10):
                        is_active = False
                    value -= datetime.timedelta(
                        microseconds=value.microseconds)
                    value = max(datetime.timedelta(seconds=0), value)
                    value = '- %s' % value.__str__()
            row.append(value)
        elif key in params:
            row.append(params[key])
        elif dfi is not None and key in dfi:
            if isinstance(dfi[key], float):
                value = '%.3f' % dfi[key]
                min_value = df[key].min()
                max_value = df[key].max()
                row.append('%.3f< %s <%.3f' % (min_value, value, max_value))
            else:
                row.append(dfi[key])
        else:
            row.append('<none>')
    return row, is_active, None


def _summarize_log(args):
    return summarize_log(*args)


def summarize_logs(logs_dir, keys, target_key, objective, show_active):
    assert objective in ['min', 'max']
    assert target_key in keys

    args_list = []
    for name in os.listdir(logs_dir):
        args_list.append((
            logs_dir,
            name,
            keys,
            target_key,
            objective,
            show_active,
        ))

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(_summarize_log, args_list)

    rows = []
    ignored = []
    for row, is_active, log_dir_ignored in results:
        if log_dir_ignored:
            ignored.append(log_dir_ignored)
            continue
        if show_active:
            if is_active:
                rows.append(row)
        else:
            rows.append(row)

    rows = sorted(rows, key=lambda x: x[0], reverse=True)
    print(tabulate.tabulate(rows, headers=keys,
                            floatfmt='.3f', tablefmt='grid',
                            numalign='center', stralign='center',
                            showindex=True, disable_numparse=True))

    if not ignored:
        return

    print('Ignored logs:')
    for log_dir in ignored:
        print('  - %s' % log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--active', action='store_true')
    args = parser.parse_args()

    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(here, 'logs')
    print('# logs_dir = %s' % logs_dir)

    keys = [
        'name',
        'last_time',
        # 'dataset',
        # 'git_hash',
        # 'hostname',
        'model',
        # 'freeze',
        'epoch',
        'iteration',
        'validation/main/miou',
    ]
    summarize_logs(logs_dir, keys, target_key=keys[-1],
                   objective='max', show_active=args.active)
