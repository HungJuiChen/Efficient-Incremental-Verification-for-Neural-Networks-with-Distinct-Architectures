# Efficient Incremental Verification for Neural Networks with Distinct Architectures

Reference: https://github.com/uiuc-arc/Incremental-DNN-Verification

This project extends the official [Incremental Neural Network Verifiers](https://github.com/uiuc-arc/Incremental-DNN-Verification?tab=readme-ov-file) by adding an experimental framework for cross-architecture proof transfer, enabling robustness verification across networks with different hidden layer widths. Please visit the official website for set up.



</p>
</details>

<details><summary> Cross-architecture proof-transfer (ECE584 Project) </summary>
<p>


This script demonstrates **incremental verification with architecture change**,  
as described in Section 5 of the final report. We compare baseline verification  
(256×2) against widened networks (260×2, 300×2, 356×2), using IVAN proof-transfer  
and Gale–Shapley neuron mapping.

## Additions in This Repository (relative to the upstream Incremental-DNN-Verification)

| File / patch                                              | Purpose                                                                                                                                         |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `cross_arch_speed.py`                                     | **Driver script** that invokes proof-transfer on a 256×2 baseline and a chosen widened network (`--arch`).                                      |
| `nnverify/proof_transfer/mapping/utils.py`                | Patched `fc_widths` to recognise parsed ONNX layers; enables neuron-mapping when layer widths differ.                                           |
| `nnverify/bnb/bnb.py`                                     | Small guard so `_.reindex()` is only called when available (prevents `AttributeError`).                                                         |
                                                                                        |
| `tools/bump_first_layer.py`                                 | Script to widen the first hidden layer of an ONNX model by appending new neurons with random weights and biases. Used to generate widened variants for proof-transfer. |


All other files mirror the original *Incremental-DNN-Verification* [layout](https://github.com/your-link-here) so unit tests and existing IVAN/IRS experiments continue to run unchanged.

This mini-demo reproduces the **incremental-verification speed-ups** reported in our ECE 584 project. It compares a 256×2 baseline with widened variants and logs timing, mapping weight, and per-property status.

---

### Run a single experiment
`python cross_arch_speed.py --arch <WIDTH> --count [H] `

where

`arch`: new hidden layer width (e.g., 260, 300, 356)

`count`: number of robustness properties to verify

### Expected console output

```
[mapping] layer-0 map size = 300 (old=256, new=300)
[mapping‑score] total weight = 247.82
...
Cross‑architecture speed‑up 256×2 → 300×2 : 1.78×
```


all the results and plot will be saved in `results`

</details>

Author: Hung-Jui Chen (hungjui2@illinois.edu), Lang Yin (langyin2@illinois.edu)

