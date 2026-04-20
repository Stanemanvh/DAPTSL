"""
ArgsParser: Argument parser for U-Shaped SSL training.
Based on DINOv2's argument parsing pattern.
"""

import argparse
from linprobe.config_setup import setup


class ArgsParser:
    """
    Argument parser for U-Shaped SSL training.
    Handles command-line arguments and configuration setup.
    """
    
    def __init__(self, description: str = "U-Shaped SSL Training"):
        """
        Initialize the argument parser.
        
        Args:
            description: Description for the argument parser
        """
        self.parser = argparse.ArgumentParser(description, add_help=True)
        self._add_arguments()
    
    def _add_arguments(self):
        """Add command-line arguments to the parser."""
        self.parser.add_argument(
            "--config-file",
            default="",
            metavar="FILE",
            help="Path to configuration file"
        )
        self.parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Whether to not attempt to resume from the checkpoint directory"
        )
        self.parser.add_argument(
            "--eval-only",
            action="store_true",
            help="Perform evaluation only"
        )
        self.parser.add_argument(
            "--eval",
            type=str,
            default="",
            help="Eval type to perform"
        )
        self.parser.add_argument(
            "opts",
            help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
            """.strip(),
            default=None,
            nargs=argparse.REMAINDER,
        )
        self.parser.add_argument(
            "--output-dir",
            "--output_dir",
            default="",
            type=str,
            help="Output directory to save logs and checkpoints"
        )
        self.parser.add_argument(
            "--wandb",
            type=str,
            default=None,
            help="Wandb project name (e.g., u_shaped_ssl_train)"
        )
        self.parser.add_argument(
            "--wandb_logdir",
            type=str,
            default="./wandb",
            help="Where the wandb log info is stored"
        )
        self.parser.add_argument(
            "--wandb_entity",
            type=str,
            default="FILL IN",
            help="Wandb entity name"
        )
        self.parser.add_argument(
            "--load_weights",
            default="",
            help="Pretrain from checkpoint (possibly other domain)"
        )
    
    def parse_args(self, args=None):
        """
        Parse command-line arguments.
        
        Args:
            args: List of arguments to parse (if None, uses sys.argv)
        
        Returns:
            Parsed arguments object
        """
        return self.parser.parse_args(args)
    
    def parse_and_setup(self, args=None):
        """
        Parse arguments and setup configuration.
        
        Args:
            args: List of arguments to parse (if None, uses sys.argv)
        
        Returns:
            cfg: Configuration object from setup()
            parsed_args: Parsed arguments object
        
        Example:
            >>> parser = ArgsParser()
            >>> cfg, args = parser.parse_and_setup()
            >>> print(cfg)
            >>> print(args.config_file)
        """
        parsed_args = self.parse_args(args)
        cfg = setup(parsed_args)
        return cfg, parsed_args
