import flax.linen as nn
import einops


class LeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, activations=False):
        if len(x.shape) > 2:
            x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if activations:
            return x
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


class ConvLeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, activations=False):
        x = nn.Conv(64, (11, 11), 4)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), 2, padding="VALID")
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if activations:
            return x
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


class LeNet5(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, activations=False):
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
        if activations:
            return x
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x
