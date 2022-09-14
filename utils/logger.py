import os
from tensorboardX import SummaryWriter
import json
import csv
import sys
import datetime

class Logger:
    def __init__(self, log_path=None, config=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        self.log_path = log_path
        if log_path is not None:
            os.makedirs(os.path.dirname(os.path.join(log_path, 'log.txt')), exist_ok=True)
            self.log_file = open(os.path.join(log_path, 'log.txt'), 'a+')
        with open(os.path.join(log_path, 'command.sh'), 'w') as f:
            f.write("CUDA_VISIBLE_DEVICES=1 python "+" ".join(sys.argv) + '\n')
            f.write('tensorboard --logdir {} --bind_all'.format(os.path.join(log_path, 'tensorboard')))
        if config is not None:
            with open(os.path.join(log_path, 'log_config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        self.summary_writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
        self.csv_writers = {}
        self.csv_files = {}

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

    def write_metrics(self, metrics, name):
        metrics = self.regular_metrics(metrics)
        for key in metrics:
            self.log("{}: {}".format(key, metrics[key]))

        if name not in self.csv_writers:
            csv_file = open(os.path.join(self.log_path, 'metrics_'+name+'.csv'), 'w')
            csv_writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            csv_writer.writeheader()
            self.csv_writers[name] = csv_writer
            self.csv_files[name] = csv_file

        self.csv_writers[name].writerow(metrics)
        self.csv_files[name].flush()

    def regular_metrics(self, metrics):
        for key in metrics:
            if isinstance(metrics[key],float):
                metrics[key] = format(metrics[key], '.4f')
        return metrics