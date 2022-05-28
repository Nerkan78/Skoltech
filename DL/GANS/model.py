import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        self.num_features = num_features
        self.embed_features = embed_features

        self.gamma_map = nn.Linear(embed_features, num_features)
        self.bias_map = nn.Linear(embed_features, num_features)
        self.base = nn.BatchNorm2d(num_features, affine=False)

        # print(f'NUM_FEATURES {num_features}')
        # print(f'EMBED_FEATURES {embed_features}')


    def forward(self, inputs, embeds):
        # print(f'-------INPUT SIZE {inputs.size()}')
        # print(f'-------EMBED SIZE {embeds.size()}')

        gamma = self.gamma_map(embeds) # TODO 
        bias = self.bias_map(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = self.base(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.upsample = upsample
        self.downsample = downsample

        self.bn_1 = nn.Identity() if not batchnorm else AdaptiveBatchNorm(in_channels, embed_channels)
        self.relu = nn.ReLU()
        self.conv_1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.bn_2 = nn.Identity() if not batchnorm else AdaptiveBatchNorm(out_channels, embed_channels)
        self.conv_2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        

        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))



    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        # print(f"in PREACT BLOCK {inputs.is_cuda} {embeds.is_cuda}")
        assert (embeds is None and not self.batchnorm) or (embeds is not None and self.batchnorm), "You should pass embeds if using AdaptiveBatchNorm"
        inputs = F.interpolate(inputs, scale_factor = 2) if self.upsample else inputs
        x = self.bn_1(inputs) if not self.batchnorm else self.bn_1(inputs, embeds)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x) if not self.batchnorm else self.bn_2(x, embeds)
        x = self.relu(x)
        x = self.conv_2(x)

        outputs = x + inputs if self.in_channels == self.out_channels else x + self.skip(inputs)
        outputs = F.interpolate(outputs, scale_factor = 0.5) if self.downsample else outputs

        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        # super(Generator, self).__init__()
        super().__init__()

        self.output_size = 4 * 2**num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition
        
        self.class_embeddings = nn.Embedding(num_classes, noise_channels)
        embed_channels = noise_channels * 2 if use_class_condition else noise_channels
        self.res_blocks = nn.ModuleList([])

        current_channels = max_channels
        for i in range(num_blocks):
            next_channels = max(min_channels, current_channels // 2)
            res_block = PreActResBlock(in_channels=current_channels,
                                        out_channels=next_channels,
                                        embed_channels=embed_channels,
                                        batchnorm= True,
                                        upsample=True,
                                        downsample=False)
            current_channels = next_channels
            self.res_blocks.append(res_block)
        factor = 2 if use_class_condition else 1
        self.initial_map = spectral_norm(nn.Linear(noise_channels * factor, max_channels * 4 * 4 ))
        self.head = nn.Sequential(nn.BatchNorm2d(next_channels),
                                nn.ReLU(),
                                spectral_norm(nn.Conv2d(next_channels, 3, kernel_size=3, padding=1)),
                                nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # TODO
        # print(f"in GENERATOR {noise.is_cuda} {labels.is_cuda}")
        label_embeddings = self.class_embeddings(labels)
        inputs = torch.cat([noise, label_embeddings], dim=1) if self.use_class_condition else noise
        x = self.initial_map(inputs)
        x = x.reshape(-1, self.max_channels, 4, 4)
        for block in self.res_blocks:
            x = block(x, inputs)
        outputs = self.head(x)
        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        # super(Discriminator, self).__init__()
        super().__init__()

        # TODO
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_projection_head = use_projection_head

        self.res_blocks = nn.ModuleList([])

        current_channels = 3
        next_channels = min_channels
        for i in range(num_blocks):
            res_block = PreActResBlock(in_channels=current_channels,
                                        out_channels=next_channels,
                                        embed_channels=None,
                                        batchnorm= False,
                                        upsample=False,
                                        downsample=True)
            current_channels = next_channels
            next_channels = min(max_channels, current_channels * 2)

            self.res_blocks.append(res_block)
        self.class_embeddings = spectral_norm(nn.Embedding(num_classes, current_channels))
        
        self.head = nn.Sequential(
                                nn.ReLU(),
                                nn.AvgPool2d(kernel_size = 4, divisor_override = 1),
                                nn.Flatten(),

        )
        self.psi = nn.Linear(current_channels, 1)


    def forward(self, inputs, labels):
        phi = inputs
        for block in self.res_blocks:
            phi = block(phi)
        # print(f'------ before sum pool phi.size()')
        phi = self.head(phi)
        # print(f'------after sum pool {phi.size()}')
        psi = self.psi(phi)
        # print(f'------PSI SIZE {psi.size()}')

        scores = psi
        scores = scores.reshape(-1, )

        if self.use_projection_head:
            y = self.class_embeddings(labels)
            # print(f'------SIZE OF Y IS {y.size()}')
            inner_product = torch.inner(phi, y).sum(dim=1)
            # print(f'------SIZE OF inner_product IS {inner_product.size()}')

            scores += inner_product

        assert scores.shape == (inputs.shape[0],)
        return scores