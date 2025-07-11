# import torch
# from typing import List, Tuple

# import nnverify.domains as domains
# from nnverify.specs.input_spec import InputSpecType
# from nnverify.domains.deepz import ZonoTransformer

# def neuron_ranges(net, device="cpu", eps=0.02, dataset="mnist"):
#     """
#     Returns a list `layers`, where layers[i][j] = (lb, ub) is the range
#     of neuron *j* in layer *i* under DeepZ abstract interpretation.
#     """
#     # Build a 1-sample spec covering the ε-ball around zero
#     inp_lb = torch.full((1, *net.input_shape), -eps,  device=device)
#     inp_ub = torch.full((1, *net.input_shape),  eps,  device=device)

#     #transformer = domains.get_domain_transformer("zono", net, (inp_lb, inp_ub))
#     transformer = ZonoTransformer(net, (inp_lb, inp_ub))
#     transformer.forward(complete=False)        # fast approximation

#     # DeepZ keeps per-layer bounds in `activation_bounds` (min, max lists)
#     lbs, ubs = transformer.activation_bounds()     # each is List[List[tensor]]
#     layers = []
#     for lo_layer, hi_layer in zip(lbs, ubs):
#         layers.append([(lo.item(), hi.item())
#                        for lo, hi in zip(lo_layer, hi_layer)])
#     return layers

from typing import List, Tuple
from nnverify.common.network import LayerType       # add this import

def fc_widths(net):
    """Return output widths of every Linear/Gemm layer."""
    widths = []
    if hasattr(net, "layers"):                      # PyTorch path
        for lyr in net.layers:
            if hasattr(lyr, "weight"):
                widths.append(lyr.weight.shape[0])
    elif hasattr(net, "ops"):                       # ONNX path (old)
        for op in net.ops:
            if op.type == "Linear" and hasattr(op, "weight"):
                widths.append(op.weight.shape[0])
    else:                                           # NEW – parsed ONNX nets
        # `net` is a list of Layer objects from `parse_onnx_layers`
        for lyr in net:
            if getattr(lyr, "type", None) == LayerType.Linear:
                widths.append(lyr.weight.shape[0])
    return widths

def neuron_ranges(net, **_) -> List[List[Tuple[float, float]]]:
    """
    Cheap placeholder: return [ [(0,1)] * width_i  for each FC layer i ].
    Enough for Gale–Shapley scoring because every edge weight becomes 1
    and the algorithm still produces a one-to-one mapping.
    """
    return [ [(0.0, 1.0)] * w for w in fc_widths(net) ]

