# Interactive classification metrics
Get an intuitive sense for the ROC curve and other binary classification metrics with interactive visualization.

![example animation](https://github.com/davhbrown/db-lake/blob/dd6b1b7694e2c0fb1be2e6ee9e656bb80ed357c3/interactive_metrics.gif)

This is a teaching and understanding tool. Change the statistics of the normal distributions or the classification threshold to see how it affects different classification metrics. [Read the blog post](https://www.glidergrid.xyz/post-archive/understanding-the-roc-curve-and-beyond) for more information.

<sub><sup>*</sup> Matthew's Correlation Coefficient (MCC) represented as unit-normalized MCC as in [Cao et al. 2020](https://arxiv.org/abs/2006.11278).</sub>

## Install & Run

### From PyPI
Create a dedicated python environment (recommended).
```bash
python3 -m pip install interactive-classification-metrics
```
Run with Bokeh server locally from the command line:
```bash
run-app
```
Opens a web browser where you can use the application.

### By cloning the repo
1. Clone this repo `git clone https://github.com/davhbrown/interactive_classification_metrics.git`
1. `cd interactive_classification_metrics`
1. Create a dedicated python environment is recommended
1. `pip install -r requirements.txt`

Run with Bokeh server locally from the command line:
```bash
bokeh serve --show serve.py
```

Opens a web browser where you can use the application.

## Inspired by
- **Cao C, Chicco D, Hoffman MM (2020) The MCC-F1 curve: a performance evaluation technique for binary classification. [
https://doi.org/10.48550/arXiv.2006.11278](
https://doi.org/10.48550/arXiv.2006.11278)**
- [arthurcgusmao](https://github.com/arthurcgusmao), the author of [py-mcc-f1](https://github.com/arthurcgusmao/py-mcc-f1) used here
- Chicco D, Tötsch N, Jurman G (2021) The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation. [_BioData Mining_ 14:13, 1-22.](https://doi.org/10.1186/s13040-021-00244-z)
- Chicco D, Jurman G (2023) The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification. [_BioData Mining_ 16:4, 1-23.](https://doi.org/10.1186/s13040-023-00322-4)
- **the spirit of [this tweet](https://twitter.com/adad8m/status/1474754752193830912?t=NBSL0j_DSfBDQfag39YpbQ&s=19)**

## Acknowledgments
Special thanks to Dr. Davide Chicco ([@davidechicco](https://github.com/davidechicco)) for valuable feedback on this project.
