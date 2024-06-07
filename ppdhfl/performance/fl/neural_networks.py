import jax.numpy as jnp
import flax.linen as nn
import einops


class FCN(nn.Module):
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for l in range(round(10 * self.pd)):
            x = nn.Dense(round(1000 * self.pw), name=f"Dense{l}")(x)
            x *= self.scale  # This goes before activation or batch norms, and is used by heterofl
            x = nn.relu(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class CNN(nn.Module):
    "A network based on the VGG16 architecture"
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        for l in range(5):
            if l < round(5 * self.pd):
                x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_1")(x)
                x = x * self.scale
                x = nn.relu(x)
                x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_2")(x)
                x = x * self.scale
                x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(round(128 * self.pw), name="Dense1")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(round(128 * self.pw), name="Dense2")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


# DenseNet based architecture
class DenseNet121(nn.Module):
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), (2, 2), padding='VALID', use_bias=False, name="conv1/conv")(x)
        x *= self.scale
        x = nn.LayerNorm(epsilon=1.001e-5, use_bias=False, name="conv1/ln")(x)
        x = nn.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = DenseBlock(6, self.pw, self.pd, scale=self.scale, name="conv2")(x, train)
        x = TransitionBlock(0.5, name="pool2", scale=self.scale)(x, train)
        x = DenseBlock(12, self.pw, self.pd, scale=self.scale, name="conv3")(x, train)
        x = TransitionBlock(0.5, name="pool3", scale=self.scale)(x, train)
        x = DenseBlock(24, self.pw, self.pd, scale=self.scale, name="conv4")(x, train)
        x = TransitionBlock(0.5, name="pool4", scale=self.scale)(x, train)
        x = DenseBlock(16, self.pw, self.pd, scale=self.scale, name="conv5")(x, train)
        x = nn.LayerNorm(epsilon=1.001e-5, use_bias=False, name="ln")(x)
        x = nn.relu(x)
        x = einops.reduce(x, "b w h d -> b d", "mean")  # Global average pooling
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class ConvBlock(nn.Module):
    growth_rate: int
    name: str
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        x1 = nn.LayerNorm(epsilon=1.001e-5, use_bias=False, name=self.name + '_0_ln')(x)
        x1 = nn.relu(x1)
        x1 = nn.Conv(4 * self.growth_rate, (1, 1), padding='VALID', use_bias=False, name=self.name + '_1_conv')(x1)
        x1 *= self.scale
        x1 = nn.LayerNorm(epsilon=1.001e-5, use_bias=False, name=self.name + '_1_ln')(x1)
        x1 = nn.relu(x1)
        x1 = nn.Conv(self.growth_rate, (3, 3), padding='SAME', use_bias=False, name=self.name + '_2_conv')(x1)
        x1 *= self.scale
        x = jnp.concatenate((x, x1), axis=3)
        return x


class TransitionBlock(nn.Module):
    reduction: float
    name: str
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.LayerNorm(epsilon=1.001e-5, use_bias=False, name=self.name + '_ln')(x)
        x = nn.relu(x)
        x = nn.Conv(int(x.shape[3] * self.reduction), (1, 1), padding='VALID', use_bias=False, name=self.name + '_conv')(x)
        x *= self.scale
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseBlock(nn.Module):
    blocks: list[int]
    name: str
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        for i in range(round(self.blocks * self.pd)):
            x = ConvBlock(round(32 * self.pw), name=f"{self.name}_block{i + 1}", scale=self.scale)(x, train)
        return x
