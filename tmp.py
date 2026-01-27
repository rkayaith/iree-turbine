import torch

from iree.turbine.kernel.boo.op_exports.conv import ConvSignature
from iree.turbine.kernel.boo.driver.launch import get_launchable

fwd_sig = ConvSignature(
    input_shape=[2, 16, 32, 3],
    kernel_shape=[10, 2, 2, 3],
    shared_layout="NHWC",
)

wrw_sig = ConvSignature(
    input_shape=[2, 16, 32, 3],
    kernel_shape=[10, 2, 2, 3],
    shared_layout="NHWC",
    # Can specify a mode "fwd" "bwd" "wrw" with:
    mode="wrw",
)

conv_fwd = get_launchable(fwd_sig)
conv_wrw = get_launchable(wrw_sig)

torch_device = torch.device("cuda:0") if torch.cuda.is_available() else None

x, w = fwd_sig.get_sample_args(device=torch_device, seed=10)

y = conv_fwd(x, w)

# get a random dLdy to back-prop.
dLdy, _ = wrw_sig.get_sample_args(device=torch_device, seed=2)
dLdw = conv_wrw(dLdy, x)
