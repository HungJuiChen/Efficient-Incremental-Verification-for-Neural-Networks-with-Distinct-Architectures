# file: run_iv_incremental.py
import nnverify.proof_transfer.proof_transfer as pt
from nnverify import config
from nnverify.common import Domain
from nnverify.bnb import Split
from nnverify.common.dataset import Dataset
from nnverify.proof_transfer.pt_types import IVAN

class NoApprox:
    """Return the *parsed* network object unchanged."""
    def approximate(self, net_name, dataset, *_, **__):
        from nnverify import util           # local import keeps deps minimal
        return util.get_net(net_name, dataset)

# baseline run on the 256x2 network
args = pt.TransferArgs(
        net=config.MNIST_FFN_L2,           # "mnist-net_256x2.onnx"
        domain=Domain.LP,
        approx=NoApprox(), 
        dataset=Dataset.MNIST,
        eps=0.02,
        split=Split.RELU_ESIP_SCORE,
        count= 10,
        pt_method=IVAN(0.003, 0.003),
        timeout=200)

print("\n=== 1. Original architecture (256 × 2) ===")
pt.proof_transfer(args)                    # fills the template store

# now point to the wider network
args.set_net("mnist-net_260x2.onnx")       # new file in the same folder

print("\n=== 2. Wider architecture (260 × 2) ===")
pt.proof_transfer(args)                    # triggers Gale–Shapley mapping

 # now point to the wider network
# args.set_net("mnist-net_300x2.onnx")       # new file in the same folder

# print("\n=== 2. Wider architecture (300 × 2) ===")
# pt.proof_transfer(args)                    # triggers Gale–Shapley mapping

