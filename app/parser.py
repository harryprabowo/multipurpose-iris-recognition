import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('pos_arg', type=int,
                    help='A required integer positional argument')

# Optional positional argument
parser.add_argument('opt_pos_arg', type=int, nargs='?',
                    help='An optional integer positional argument')

# Optional argument
parser.add_argument('--opt_arg', type=int,
                    help='An optional integer argument')

# Switch
parser.add_argument('--switch', action='store_true',
                    help='A boolean switch')
