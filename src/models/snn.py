"""
Spiking Neural Network for Event Detection
Energy-efficient neuromorphic computing for real-time EEG analysis
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class SNNConfig:
    """Configuration for Spiking Neural Network"""
    input_size: int = 256
    hidden_sizes: tuple = (128, 64)
    output_size: int = 16
    beta: float = 0.95  # Membrane potential decay rate
    threshold: float = 1.0  # Spike threshold
    num_steps: int = 25  # Time steps


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire Neuron"""

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_current, mem):
        """
        Args:
            input_current: Input current (batch, features)
            mem: Membrane potential (batch, features)
        Returns:
            spike, mem
        """
        mem = self.beta * mem + input_current
        spike = (mem > self.threshold).float()
        mem = mem * (1 - spike)  # Reset
        return spike, mem


class SpikingNeuralNetwork(nn.Module):
    """
    Spiking Neural Network for event-based EEG processing
    Low-power, neuromorphic architecture for real-time analysis
    """

    def __init__(self, config: Optional[SNNConfig] = None):
        super().__init__()
        self.config = config or SNNConfig()

        # Build layers
        layers = []
        neurons = []

        input_size = self.config.input_size
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            neurons.append(LIFNeuron(self.config.beta, self.config.threshold))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.config.output_size))
        neurons.append(LIFNeuron(self.config.beta, self.config.threshold))

        self.layers = nn.ModuleList(layers)
        self.neurons = nn.ModuleList(neurons)

    def forward(self, x):
        """
        Forward pass through SNN

        Args:
            x: Input tensor (batch, input_size)

        Returns:
            Spike output over time (batch, output_size)
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize membrane potentials
        mem = [torch.zeros(batch_size, layer.out_features, device=device)
               for layer in self.layers]

        # Accumulate spikes
        spike_outputs = []

        # Simulate over time steps
        for t in range(self.config.num_steps):
            layer_input = x

            for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
                current = layer(layer_input)
                spike, mem[i] = neuron(current, mem[i])
                layer_input = spike

            spike_outputs.append(spike)

        # Sum spikes over time
        output = torch.stack(spike_outputs).sum(dim=0)

        return output
