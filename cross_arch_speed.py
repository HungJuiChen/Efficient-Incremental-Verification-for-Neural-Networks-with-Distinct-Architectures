import argparse
from nnverify import config
from nnverify.common import Domain
from nnverify.bnb import Split
from nnverify.common.dataset import Dataset
from nnverify.proof_transfer.pt_types import IVAN
import nnverify.proof_transfer.proof_transfer as pt

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=int, default=300, help='Target architecture width (e.g., 260 or 300)')
parser.add_argument('--count', type=int, default=50, help='Number of properties to test')
args_cli = parser.parse_args()

# Choose ONNX file based on architecture
class LoadTargetNet:
    def approximate(self, net_name, dataset, *_, **__):
        from nnverify import util
        return util.get_net(f"nnverify/nets/mnist-net_{args_cli.arch}x2.onnx", dataset)

# Set up arguments
args = pt.TransferArgs(
        net=config.MNIST_FFN_L2,     # 256×2 baseline
        domain=Domain.LP,
        approx=LoadTargetNet(),      # <-- dynamically selected target net
        dataset=Dataset.MNIST,
        eps=0.02,
        split=Split.RELU_ESIP_SCORE,
        count=args_cli.count,
        pt_method=IVAN(0.003, 0.003),
        timeout=600)

# Run transfer and report
speed = pt.proof_transfer(args)
print(f"\nCross-architecture speed-up 256×2 → {args_cli.arch}×2 :  {speed:4.2f}×")
