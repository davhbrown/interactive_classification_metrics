# Interactive classification metrics
Get an intuitive sense for the ROC curve and other binary classification metrics with interactive visualization.

![example image](interactive_metrics.png?raw=true)

## Install & Run
1. Clone this repo `git clone https://github.com/davhbrown/interactive-classification-metrics.git`
1. `cd interactive-classification-metrics`
1. Create a new python environment if you wish. This was developed in Python 3.9.12. Python 3.11+ is not supported.
1. `pip install -r requirements.txt`

#### Run with Bokeh server locally
From the command line:
`bokeh serve --show serve.py`

## Inspired by
- **Cao C, Chicco D, Hoffman MM (2020) The MCC-F1 curve: a performance evaluation technique for binary classification. [arXiv:2006.11278](https://arxiv.org/abs/2006.11278) [stat.ML]**
- [arthurcgusmao](https://github.com/arthurcgusmao), the author of [py-mcc-f1](https://github.com/arthurcgusmao/py-mcc-f1) used here
- Chicco D, Tötsch N, Jurman G (2021) The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation. _BioData Mining_ 14:13, 1-22.
- the spirit of [this tweet](https://twitter.com/adad8m/status/1474754752193830912?t=NBSL0j_DSfBDQfag39YpbQ&s=19)
