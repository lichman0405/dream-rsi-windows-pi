# References and provenance

- Dream-RSI paper, version 2: https://arxiv.org/abs/2609.14858 . The §3 mechanism is the target of this independent reproduction. The authors' repository is https://github.com/zhengkid/Dream-RSI ; this project does not claim to contain the authors' source.
- Pi coding agent: https://github.com/earendil-works/pi . The validated live path uses Pi through its RPC interface.
- DeepSeek Harness: https://github.com/deepseek-ai/deepseek-harness . An optional adapter is present, but its live behavior was not accepted in this project's Pi-only verification.

The bundled knapsack task is a small software test case, not a paper benchmark. The [MOF experiment design](experiments/MOF_SCREENING_DESIGN.md) proposes the public CRAFTED **2.0.1** simulated adsorption dataset (version DOI: [10.5281/zenodo.10120180](https://doi.org/10.5281/zenodo.10120180)). Its MOF subset has 690 materials; the earlier 726-material release at DOI 10.5281/zenodo.7106174 is not the proposed input. Dataset component licenses are listed in the design and are separate from the code license. The experiment has not been implemented or run; no MOF performance result is claimed here.
