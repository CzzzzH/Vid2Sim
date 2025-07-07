import torch 
from torch import nn 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, dim, output_dim=128):
        super(PositionalEncoding, self).__init__()
        self.output_dim = output_dim
        self.num_coordinates = dim  # For 3D positions

        # Calculate the number of frequency bands per coordinate
        total_frequency_bands = output_dim // (2 * self.num_coordinates)
        frequencies = 2 ** torch.linspace(0, total_frequency_bands - 1, total_frequency_bands)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):

        x_expanded = x.unsqueeze(-1) * self.frequencies  # Shape: (..., 3, L)
        x_sin = torch.sin(x_expanded)
        x_cos = torch.cos(x_expanded)
        encoding = torch.cat([x_sin, x_cos], dim=-1)  # Shape: (..., 3, 2L)
        encoding = encoding.view(*x.shape[:-1], -1)  # Shape: (..., 3 * 2L)

        current_dim = encoding.shape[-1]
        if current_dim > self.output_dim:
            encoding = encoding[..., :self.output_dim]
        elif current_dim < self.output_dim:
            pad_width = self.output_dim - current_dim
            padding = torch.zeros(*encoding.shape[:-1], pad_width, device=x.device)
            encoding = torch.cat([encoding, padding], dim=-1)

        return encoding

class ResidualBlock(nn.Module):

    def __init__(self, layer_width):
        super(ResidualBlock, self).__init__()
        self.layer_width = layer_width
        self.linear1 = nn.Linear(layer_width, layer_width)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(layer_width, layer_width)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        return x + residual

class SimplicitsMLP(nn.Module):
    
    def __init__(self, spatial_dimensions, layer_width, num_handles, num_layers):
        super(SimplicitsMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(spatial_dimensions, layer_width))
        layers.append(nn.ELU())

        for i in range(num_layers):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ELU())
        
        layers.append(nn.Linear(layer_width, num_handles))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x.clone())
        return output

class JacobianMLP(nn.Module):
    
    def __init__(self, spatial_dimensions, layer_width, num_handles, num_layers):
        super(JacobianMLP, self).__init__()
        
        self.positional_encoding = PositionalEncoding(dim=spatial_dimensions, output_dim=layer_width // 2)

        self.layers_1 = nn.Sequential(*[
            ResidualBlock(layer_width // 2),
            nn.Linear(layer_width // 2, layer_width),
            nn.GELU(),
        ])
        self.layers_2 = nn.Sequential(*[
            ResidualBlock(layer_width),
            ResidualBlock(layer_width)
        ])

        self.final_layer = nn.Linear(layer_width, 9*12*num_handles)

    def forward(self, x):
        x_embedding = self.positional_encoding(x)
        output = self.layers_1(x_embedding)
        output = self.layers_2(output)
        output = self.final_layer(output)
        return output 