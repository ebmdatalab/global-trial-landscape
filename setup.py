import argparse
import logging
import pathlib
from os import environ


def setup_logger(verbosity):
    logging_level = logging.ERROR
    if verbosity > 0:
        logging_level = logging.WARNING
    if verbosity > 1:
        logging_level = logging.INFO
    if verbosity > 2:
        logging_level = logging.DEBUG
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s: [%(levelname)-9s]: %(message)s [%(module)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class MissingEnvironmentVariable(Exception):
    pass


def get_env_setting(setting):
    """Get the environment setting or return exception"""
    try:
        return environ[setting]
    except KeyError:
        error_msg = f"Set the {setting} env variable"
        raise MissingEnvironmentVariable(error_msg)


def get_verbosity_parser():
    verb = argparse.ArgumentParser(add_help=False)
    verb.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
    )
    return verb


def get_base_parser():
    verb = get_verbosity_parser()
    base = argparse.ArgumentParser(add_help=False, parents=[verb])
    base.add_argument(
        "--input-file",
        required=True,
        type=pathlib.Path,
        help="Path to input file",
    )
    return base


def get_full_parser():
    base = get_base_parser()
    parent = argparse.ArgumentParser(add_help=False, parents=[base])
    parent.add_argument(
        "--output-dir", type=pathlib.Path, help="Alternate directory to store results"
    )
    parent.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Break queries into chunk-size chunks",
    )
    return parent
