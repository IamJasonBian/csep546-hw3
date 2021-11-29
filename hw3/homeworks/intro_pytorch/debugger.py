from layers import LinearLayer
import torch


generator = torch.Generator()
generator.manual_seed(446)

layer = LinearLayer(20, 10, generator=generator)
x = torch.ones((5, 20))

actual = layer(x)
torch.testing.assert_allclose(actual.shape, (5, 10))


