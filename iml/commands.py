"""
Commands
"""

import sys
import argparse

from flask_cors import CORS

from iml.server import app
from iml import Config
# from iml.config import mode


# def start(env='development'):
#     if env == 'development':
#         app.run()
#     elif env == 'production':
#         pass


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Command Line Tools for running RNNVis')
    parser.add_argument('method', choices=['start'],
                        help='run iml start to start the server')
    parser.add_argument('--prod', '-p', dest='prod', action='store_const', const=True, default=False,
                        help='set this flag to run production environment')
    # parser.add_argument('--force', '-f', dest='force', action='store_const', const=True, default=False,
    #                     help='set this flag to force re-seed db')
    parser.add_argument('--debug', '-d', dest='debug', action='store_const', const=True, default=False,
                        help='set this flag to set debug mode for flask app')
    args = parser.parse_args(args)

    if args.method == 'start':
        Config.mode('development')
        CORS(app)
        app.run(debug=args.debug)
        # start(env='production' if args.prod else 'development')
    # elif args.method == 'seeddb':
    #     seed_db(args.force)
    #     print("Seeding Done.")


if __name__ == '__main__':
    main()