# Citation and Attribution

This repository was written for the project **Riemannian Optimizers for Hyperbolic WordNet Subtree Classification**. If you reuse the code or results, cite the accompanying report:

```bibtex
@misc{sjiang2026riemannian_wordnet,
  title  = {Riemannian Optimizers for Hyperbolic WordNet Subtree Classification},
  author = {Sijia Jiang},
  year   = {2026},
  note   = {Course project report and code repository}
}
```

## External Methods, Data, and Libraries

The project builds on the following work and software:

```bibtex
@inproceedings{nickel2017poincare,
  title     = {Poincare Embeddings for Learning Hierarchical Representations},
  author    = {Nickel, Maximilian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@inproceedings{ganea2018hyperbolic,
  title     = {Hyperbolic Neural Networks},
  author    = {Ganea, Octavian-Eugen and Becigneul, Gary and Hofmann, Thomas},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2018}
}

@inproceedings{becigneul2019riemannian,
  title     = {Riemannian Adaptive Optimization Methods},
  author    = {Becigneul, Gary and Ganea, Octavian-Eugen},
  booktitle = {International Conference on Learning Representations},
  year      = {2019}
}

@article{bonnabel2013stochastic,
  title   = {Stochastic Gradient Descent on Riemannian Manifolds},
  author  = {Bonnabel, Silvere},
  journal = {IEEE Transactions on Automatic Control},
  year    = {2013}
}

@article{kochurov2020geoopt,
  title   = {Geoopt: Riemannian Optimization in PyTorch},
  author  = {Kochurov, Max and Karimov, Rasul and Kozlukov, Serge},
  journal = {arXiv preprint arXiv:2005.02819},
  year    = {2020}
}

@article{miller1995wordnet,
  title   = {WordNet: A Lexical Database for English},
  author  = {Miller, George A.},
  journal = {Communications of the ACM},
  year    = {1995}
}
```

The optional embedding-training stage uses the public `facebookresearch/poincare-embeddings` implementation. That repository is intentionally not vendored here; only converted CSV embeddings are included for reproducibility of the classifier experiments.

The implementation also depends on PyTorch, NumPy, pandas, scikit-learn, NetworkX, Matplotlib, PyYAML, tqdm, NLTK, and Geoopt. See `requirements.txt`.
