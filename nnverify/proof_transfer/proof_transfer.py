from nnverify import config
from nnverify.analyzer import Analyzer
from nnverify.common.result import Result, Results
from nnverify.proof_transfer.pt_util import compute_speedup, plot_verification_results
from nnverify.proof_transfer.pt_types import ProofTransferMethod
from nnverify.proof_transfer.mapping import layer_match, neuron_ranges


class TransferArgs:
    def __init__(self, domain, approx, pt_method=None, count=None, eps=0.01, dataset='mnist', attack='linf',
                 split=None, net='',
                 timeout=30):
        self.net = config.NET_HOME + net
        self.domain = domain
        self.pt_method = pt_method
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.attack = attack
        self.split = split
        self.approximation = approx
        self.timeout = timeout

    def set_net(self, net):
        self.net = config.NET_HOME + net

    def get_verification_arg(self):
        arg = config.Args(net=self.net, domain=self.domain, dataset=self.dataset, eps=self.eps,
                          split=self.split, count=self.count, pt_method=self.pt_method, timeout=self.timeout)
        # net is set correctly again since the home dir is added here
        arg.net = self.net
        return arg


def proof_transfer(pt_args):
    res, res_pt = proof_transfer_analyze(pt_args)

    speedup = compute_speedup(res, res_pt, pt_args)
    print("Proof Transfer Speedup :", speedup)
    plot_verification_results(res, res_pt, pt_args)
    return speedup


def proof_transfer_acas(pt_args):
    res = Results(pt_args)
    res_pt = Results(pt_args)
    for i in range(1, 6):
        for j in range(1, 10):
            pt_args.set_net(config.ACASXU(i, j))
            pt_args.count = 4
            r, rp = proof_transfer_analyze(pt_args)
            res.results_list += r.results_list
            res_pt.results_list += rp.results_list

    # compute merged stats
    res.compute_stats()
    res_pt.compute_stats()

    speedup = compute_speedup(res, res_pt, pt_args)
    print("Proof Transfer Speedup :", speedup)
    plot_verification_results(res, res_pt, pt_args)
    return speedup

###added by hungjui2
# helper to count neurons for Linear layers
def fc_widths(net):
    """
    Return a list of output widths for every Linear (Gemm) layer
    regardless of backend (torch or onnx).
    """
    from nnverify.common.network import LayerType
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



def proof_transfer_analyze(pt_args):
    args = pt_args.get_verification_arg()
    analyzer = Analyzer(args)
    _ = analyzer.run_analyzer()
    template_store = analyzer.template_store
    if args.pt_method == ProofTransferMethod.ALL:
        # precomputes reordered template store
        template_store = get_reordered_template_store(args, template_store)
    approx_net = get_perturbed_network(pt_args)
    ### added by hungjui2
    # ---------- OPTIONAL mapping when layer widths differ ----------
    # if hasattr(analyzer.net, "layers") and hasattr(approx_net, "layers"):
    #     same_arch = all(layer_width(l0) == layer_width(l1)
    #                     for l0, l1 in zip(analyzer.net.layers,
    #                                       approx_net.layers))
    # else:
    #     same_arch = True          # ONNX wrappers in PLDI tests
    #print("[probe] top-level attrs on approx_net:", dir(approx_net)[:20])
    old_w = fc_widths(analyzer.net)
    new_w = fc_widths(approx_net)
    print("[debug] old widths:", old_w)
    print("[debug] new widths:", new_w)

    same_arch = (old_w == new_w and len(old_w) > 0)

    if not same_arch:
        old_ranges = neuron_ranges(analyzer.net)
        new_ranges = neuron_ranges(approx_net)
        maps = [layer_match(a, b) for a, b in zip(old_ranges, new_ranges)]
        if any(maps):                # at least one layer had a mapping
            args.neuron_map = maps
            print(f"[mapping] layer-0 map size = {len(maps[0])} "
                f"(old={old_w[0]}, new={new_w[0]})")
        else:
            print("[mapping] **mapping empty – skipped**")    
    # -----------------------------------------------------------------

    # Use template generated from original verification for faster verification of the approximate network
    approx_args = pt_args.get_verification_arg()
    res_pt = Analyzer(approx_args, net=approx_net, template_store=template_store).run_analyzer()
    # Compute results without any template store as the baseline
    res = Analyzer(args, net=approx_net).run_analyzer()
    return res, res_pt


def get_reordered_template_store(args, template_store):
    args.pt_method = ProofTransferMethod.REORDERING
    # Compute reordered template
    analyzer_reordering = Analyzer(args, template_store=template_store)
    # TODO: make this as a separate a function from run_analyzer that takes some budget for computing
    _ = analyzer_reordering.run_analyzer()
    # This template store should contain leaf nodes of reordered tree
    template_store = analyzer_reordering.template_store
    return template_store


def get_perturbed_network(pt_args):
    # Generate the approximate network
    # approx_net = pt_args.approximation.approximate(pt_args.net, pt_args.dataset)
    # return approx_net
    #add by hungjui2
    if pt_args.approximation is None:
        return None
    return pt_args.approximation.approximate(pt_args.net, pt_args.dataset)
