#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from ibi.ibi_method import ibi_method, clean_folders
from ibi import plotting
import logging


class bibi_main():
    def __init__(self):
        parser = ArgumentParser(usage='bibi.py <command> [<args>]')
        parser.add_argument('command', choices=('run', 'restart', 'plot', 'clean'))
        args = parser.parse_args(sys.argv[1:2])
        getattr(self, args.command)()


    def run(self):
        init_logging()
        parser = ArgumentParser(
                description='Runs IBI fitting from scratch')
        parser.add_argument('ini', nargs='?', default='ibi.ini')
        args = parser.parse_args(sys.argv[2:])
        method = ibi_method(args.ini, clean_start=True)
        method.run_iterations()


    def restart(self):
        init_logging(file_mode='a')
        parser = ArgumentParser(
                description='Restats IBI fitting from last viable iteration')
        parser.add_argument('ini', nargs='?', default='ibi.ini')
        args = parser.parse_args(sys.argv[2:])
        method = ibi_method(args.ini, clean_start=False)
        method.run_iterations()


    def plot(self):
        parser = ArgumentParser(
                description='Plot potentials and distributions')
        parser.add_argument('mode', choices=plotting.plot_modes)
        parser.add_argument('--iteration', default=0, type=int)
        parser.add_argument('--ini', '-i', nargs='?', default='ibi.ini')
        args = parser.parse_args(sys.argv[2:])
        plotting.plot_cmd(args)


    def clean(self):
        parser = ArgumentParser(
                description='Clean current folder from previous results')
        clean_folders()


def init_logging(file_mode='w'):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt='%H:%M:%S:',
        handlers=[
            logging.FileHandler('ibi.log', mode=file_mode),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    bibi_main()

