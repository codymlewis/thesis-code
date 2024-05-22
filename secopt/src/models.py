import jax.numpy as jnp
import flax.linen as nn
import einops


class Small(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class CNN(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = nn.Conv(48, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class LeNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = nn.Conv(6, (5, 5), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))
        x = nn.Conv(16, (5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


# ConvNext
class ConvNeXt(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, representation=False):
        depths = [3, 3, 27, 3]
        projection_dims = [128, 256, 512, 1024]
        # Stem block.
        stem = nn.Sequential([
            nn.Conv(projection_dims[0], (4, 4), strides=(4, 4), name="convnext_base_stem_conv"),
            nn.LayerNorm(epsilon=1e-6, name="convnext_base_stem_layernorm"),
        ])

        # Downsampling blocks.
        downsample_layers = [stem]

        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            downsample_layer = nn.Sequential([
                nn.LayerNorm(epsilon=1e-6, name=f"convnext_base_downsampling_layernorm_{i}"),
                nn.Conv(projection_dims[i + 1], (2, 2), strides=(2, 2), name=f"convnext_base_downsampling_conv_{i}"),
            ])
            downsample_layers.append(downsample_layer)

        num_convnext_blocks = 4
        for i in range(num_convnext_blocks):
            x = downsample_layers[i](x)
            for j in range(depths[i]):
                x = ConvNeXtBlock(
                    projection_dim=projection_dims[i],
                    layer_scale_init_value=1e-6,
                    name=f"convnext_base_stage_{i}_block_{j}",
                )(x)

        x = einops.reduce(x, 'b h w c -> b c', 'mean')
        if representation:
            return x
        x = nn.LayerNorm(epsilon=1e-6, name="convnext_base_head_layernorm")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class ConvNeXtBlock(nn.Module):
    projection_dim: int
    name: str = None
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        x = nn.Conv(
            self.projection_dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.projection_dim,
            name=self.name + "_depthwise_conv",
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, name=self.name + "_layernorm")(x)
        x = nn.Dense(4 * self.projection_dim, name=self.name + "_pointwise_conv_1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.projection_dim, name=self.name + "_pointwise_conv_2")(x)

        if self.layer_scale_init_value is not None:
            x = LayerScale(
                self.layer_scale_init_value,
                self.projection_dim,
                name=self.name + "_layer_scale",
            )(x)

        return inputs + x


class LayerScale(nn.Module):
    init_values: float
    projection_dim: int
    name: str
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            'gamma',
            nn.initializers.constant(self.init_values, dtype=self.param_dtype),
            (self.projection_dim,),
            self.param_dtype,
        )
        return x * gamma


# ResNetV2
class ResNetV2(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="VALID", use_bias=True, name="conv1_conv")(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = Stack2(64, 3, name="conv2")(x)
        x = Stack2(128, 4, name="conv3")(x)
        x = Stack2(256, 6, name="conv4")(x)
        x = Stack2(512, 3, strides1=(1, 1), name="conv5")(x)

        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = einops.reduce(x, "b h w d -> b d", 'mean')

        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        return nn.softmax(x)


class Block2(nn.Module):
    filters: int
    kernel: (int, int) = (3, 3)
    strides: (int, int) = (1, 1)
    conv_shortcut: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        preact = nn.LayerNorm(epsilon=1.001e-5)(x)
        preact = nn.relu(preact)

        if self.conv_shortcut:
            shortcut = nn.Conv(
                4 * self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_0_conv"
            )(preact)
        else:
            shortcut = nn.max_pool(x, (1, 1), strides=self.strides) if self.strides > (1, 1) else x

        x = nn.Conv(
            self.filters, (1, 1), strides=(1, 1), padding="VALID", use_bias=False,
            name=self.name + "_1_conv"
        )(preact)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.Conv(
            self.filters, self.kernel, strides=self.strides, padding="VALID", use_bias=False,
            name=self.name + "_2_conv"
        )(x)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(4 * self.filters, (1, 1), name=self.name + "_3_conv")(x)
        x = shortcut + x
        return x


class Stack2(nn.Module):
    filters: int
    blocks: int
    strides1: (int, int) = (2, 2)
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = Block2(self.filters, conv_shortcut=True, name=self.name + "_block1")(x)
        for i in range(2, self.blocks):
            x = Block2(self.filters, name=f"{self.name}_block{i}")(x)
        x = Block2(self.filters, strides=self.strides1, name=f"{self.name}_block{self.blocks}")(x)
        return x
