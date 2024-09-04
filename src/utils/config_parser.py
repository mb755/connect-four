"""!@package config_parser
@brief Module creating a parser for default command line arguments

@details
The default config_parser object can be further customized as needed.

For function details see the function documentation:
- config_parser.py

@author Mate Balogh
@date 2024-09-04
"""

import argparse
import os

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def default_parser(description):
    """!@brief creates a parser for default command line arguments
    @details
    The default arguments are: <br>
    -o, --output-suffix: string to be appended to all output filenames <br>

    @param description (str): text that is displayed in the -h help output

    @return config parser: command line argument parser
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--output-suffix",
        help="This string is appended to all output filenames",
        required=False,
        default="",
        type=str,
    )
    return parser


def training_parser(description):
    """!@brief creates a parser for command line arguments used for training
    @details
    uses the default parser, but adds arguments specific to training
    see default_parser for more details <br>
    additional arguments are: <br>
    -e, --epochs: number of epochs to train for <br>
    -s, --starting-epoch: number of epochs pre-existing model was trained for; useful when resuming training

    @param description (str): text that is displayed in the -h help output

    @return config parser: command line argument parser
    """
    parser = default_parser(description)

    # these might be relevant for training
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs to train for",
        required=False,
        default=300,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--starting-epoch",
        help="number of epochs pre-existing model was trained for; useful when resuming training",
        required=False,
        default=0,
        type=int,
    )

    # optional for continuing training
    parser.add_argument(
        "-i",
        "--input-file",
        help="location of model to load",
        required=False,
        type=str,
    )
    return parser


def evaluation_parser(description):
    """!@brief creates a parser for command line arguments used for evaluation
    @details
    uses the default parser, but adds arguments specific to evaluation
    see default_parser for more details <br>
    additional arguments are: <br>
    -i, --input-file: location of model to load

    @param description (str): text that is displayed in the -h help output

    @return config parser: command line argument parser
    """
    parser = default_parser(description)

    # necessary for evaluation
    parser.add_argument(
        "-i",
        "--input-file",
        help="location of model to load",
        required=False,
        default=f"{root_dir}/output/trained_agent_weights.pth",
        type=str,
    )
    return parser
