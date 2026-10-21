"""Unmodified LoRA/Init_Weight classes extracted from the official implementation.

Copyright (c) 2024 Michelle Halbheer, Dominik J. Mühlematter. See LICENSE.
"""
import enum
from torch import nn

class Init_Weight(enum.Enum):
    NORMAL = 0
    KAIMING_UNIFORM = 1
    XAVIER_UNIFORM = 2
    DEFAULT = NORMAL

class LoRA(nn.Module):
    def __init__(
            self,
            w: nn.Module,
            rank: int,
            dim: int,
            initialize: bool = True,
            init_type: Init_Weight = Init_Weight.DEFAULT,
            init_settings: dict = None,
            out_dim: int = None
    ) -> None:
        """
        Implements a Low Rank Adaptation module

        Parameters
        ----------
        w : nn.Module
            The original projection layer
        rank : int
            The rank of the Low Rank Adaptation module
        dim : List[int]
            The dimension of the weight matrix
        initialize : bool, optional
            Whether to initialize the weights, by default True
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        out_dim : int, optional
            The output dimension of the LoRA layer, by default None
        """

        super().__init__()

        # LoRA rank
        self.rank = rank

        # Original projection layer weights
        self.w = w

        if out_dim is None:
            out_dim = dim

        # LoRA matrices
        self.w_a = nn.Linear(dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

        # Initialize the weights if needed
        if initialize:
            self.initialize_weights(init_type=init_type, init_settings=init_settings)

    def forward(self, x):
        """
        Forward pass for the LoRA Layer

        Parameters
        ----------
        x : Tensor
            The input tensor

        Returns
        -------
        out : Tensor
            The output tensor
        """

        out = self.w(x) + self.w_b(self.w_a(x))

        return out

    def initialize_weights(self, init_type: Init_Weight = Init_Weight.DEFAULT, init_settings: dict = None) -> None:
        """
        Initialize the weights of the LoRA matrices

        Parameters
        ----------
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        if init_type == Init_Weight.NORMAL:
            # Set the mean and standard deviation
            if init_settings is None:
                mean = 0
                std = 0.02
            else:
                mean = init_settings["mean"]
                std = init_settings["std"]

            # Initialize all weights separately
            nn.init.normal_(self.w_a.weight, mean=mean, std=std)
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.KAIMING_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                a_squared = 5
            else:
                a_squared = init_settings["a_squared"]

            # Initialize all weights separately
            from math import sqrt
            nn.init.kaiming_uniform_(self.w_a.weight, a=sqrt(a_squared))
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.XAVIER_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                gain  = 1
            else:
                gain = init_settings["gain"]

            # Initialize all weights separately
            nn.init.xavier_uniform_(self.w_a.weight, gain=gain)
            nn.init.zeros_(self.w_b.weight)
        else:
            raise ValueError("Invalid initialization type")
