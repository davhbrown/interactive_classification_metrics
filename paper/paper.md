---
title: 'Interactive Classification Metrics: An interactive graphical Python package for building robust intuition for classification model evaluation'
tags:
  - classification
  - machine learning
  - ROC curve
  - matthews correlation coefficient
  - Python
authors:
  - name: David H. Brown
    orcid: 0000-0002-0969-8711
    affiliation: 1
    corresponding: true
  - name: Davide Chicco
    orcid: 0000-0001-9655-7142
    affiliation: "2, 3"
affiliations:
 - name: Independent Researcher, Austin, Texas, USA
   index: 1
 - name: Dipartimento di Informatica Sistemistica e Comunicazione, Università di Milano-Bicocca, Milan, Italy
   index: 2
 - name: Institute of Health Policy Management and Evaluation, University of Toronto, Toronto, Ontario, Canada
   index: 3
date: 29 September 2024
bibliography: paper.bib
---

# Summary

Machine Learning continues to grow in popularity in academia, in industry, and
is increasingly used in other fields. However, most of the common metrics used
to evaluate even simple binary classification models have certain flaws that
are neither immediately obvious nor consistently taught to practitioners. Here
we present Interactive Classification Metrics (ICM), a Python package to
visualize and explore the relationships between different evaluation metrics.
The user changes the distribution statistics and explores corresponding
changes across a suite of evaluation metrics. The interactive, graphical
nature of this tool emphasizes the tradeoffs of each metric without the
overhead of data wrangling and model training. The goals of this application
are: (1) to aid practitioners in the ever-expanding machine learning field to
choose the most appropriate evaluation metrics for their classification
problem; (2) to promote careful attention to interpretation that is required
even in the simplest scenarios like binary classification. Our software
is publicly available for free under the MIT license on PyPI at
https://pypi.org/project/interactive-classification-metrics/ and on GitHub
at https://github.com/davhbrown/interactive_classification_metrics.

# Statement of need

More people enter the machine learning field every year through a variety of
paths. In the United States alone, “Data Scientist” job growth is expected to
outpace the average over the next decade, with similar growth in other
countries [@Kaggle:2022; @BLS:2024]. User-friendly,
low barrier to entry libraries like scikit-learn are used widely across the
industry, and fields beyond statistics increasingly use machine learning
tools, but are not always familiar with basic best practices like avoiding
data leakage, not to mention careful interpretation of evaluation metrics
`[@Liu:2019; @Kaggle:2022; @Kapoor:2023; @Checkroud:2024]`.

Classification models are a fundamental tool in machine learning. The quality
of a classification model is evaluated by comparing model predictions with
ground truth targets, forming sections of the confusion matrix, and resulting
in the True Positive Rate, True Negative Rate, Positive Predictive Value, and
Negative Predictive Value (NPV). From these four “basic rates”
`[@Chicco:2023]`, further evaluation metrics have been derived, each
summarizing different aspects of a model’s predictive performance. Common
metrics have specific caveats known to experts, but not immediately apparent
to novices, and can even be overlooked by experienced modelers. For example,
interpretation of Accuracy, Recall, and most other metrics depend on class
(im)balance and comparison to each other. While these metrics do not claim to
capture all four quadrants of the confusion matrix, in practice they are often
reported on their own or as a single number that adequately describes a
model’s predictive quality.

Literature details the shortcomings with Receiver Operating Characteristic
(ROC) curves `[@Chicco:2023]` and singular metrics `[@Powers:2020]`; has
evaluated less common metrics like bookmaker informedness and Matthews
Correlation Coefficient (MCC) `[@Chicco:2021; @Chicco:2023]`; and even
proposed novel graphical tools like the MCC-F1 curve `[@Cao:2020]`. However,
insights from scholarly work take time to enter widespread educational
material, and are not immediately obvious from static examples or mathematical
formulae alone.

It is essential that the machine learning community have every pedagogical
resource at its disposal to solidify a deep understanding of and intuition for
the field’s most basic, but nuanced, model evaluation tools. ICM aims to
improve the choices novice scientists make, and solidify those of experienced
practitioners, when building classification models.


# Overview of Functionality

After installing ICM in a Python environment, it is callable from the terminal
with a single command, which opens a web browser for interaction. It runs on
the user’s local machine using bokeh server and standard data/machine learning
Python libraries for underlying computation.

![Screenshot of the application with numbered steps overlaid (red circles). Users control 9 interactive sliders at the top, and all graphs respond accordingly. The sliders control the sample size (N), mean, standard deviation (SD), and skew of the two distributions (Step 1) that represent the negative (black) and positive class predictions (orange). The properties of these distributions, along with the classification threshold (green; Step 2) control the magnitude and shape of all other plots. Users can also choose to show or hide specific plots with checkboxes (Step 3). Full display shown.\label{fig:1}](Figure1.png)

A brief animation of the application in use can be seen on
[GitHub](https://github.com/davhbrown/interactive_classification_metrics). The
full list of plots, evaluation metrics, and their acronyms are: Class
distributions, ROC curve, Area Under the ROC curve (ROC AUC), Precision-Recall
(PR) curve, Area Under the PR curve (PR AUC), Confusion Matrix, Matthews
Correlation Coefficient-F1 (MCC-F1) curve, Accuracy, Recall, Specificity,
Precision, Negative Predictive Value (NPV), F1-Score, and MCC (unit
normalized). Different fields have different naming conventions for the same
metric; for example, Recall is also called the True Positive Rate,
Sensitivity, or Hit Rate. A comprehensive list of alternative
naming conventions is on the
[confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion)
Wikipedia page.

# Example Scenario

\autoref{fig:2} depicts ICM demonstrating a classic model evaluation mistake:
using Accuracy alone to evaluate an overfit model trained on an imbalanced
dataset. A comprehensive look at all available metrics reveal the flaw: with
additional information beyond Accuracy and ROC AUC, it is clear that when
accounting for all four “basic rates”, MCC reveals chance performance
(MCC = 0.5). Also note that PR AUC=0.9, which could be naively interpreted as
a good model, but its minimum baseline, driven by the proportion of classes,
is 0.83, indicating that overall the area covered by the PR curve is small.

![The classic flaw of Accuracy on an imbalanced dataset. The negative class (black) has N=100 examples, the positive class (orange) has N=500. The classification threshold (green) is set extremely low to represent a model that predicts everything as the positive class, yet achieves over 80% Accuracy due to the proportions of the two classes in the dataset.\label{fig:2}](Figure2.png)

# Conclusion
Visualizing results is a pivotal task in many fields, and scientific
visualization is a cornerstone of modern data science `[@Midway:2020]`.
Software tools can make this task easier for researchers and students, but to
the best of our knowledge, no Python package for visualizing binary
classification results in an interactive, exploratory way exists. We fill this
gap by providing our ICM software. Our application does not require the
overhead of cleaning data and training models. Instead, it allows users to
focus on understanding the evaluation metrics themselves, and how these
metrics change in relation to each other as the user alters the statistics of
the two distributions.

# Acknowledgements

The authors thank Arthur Colombini Gusmão, the author of the `py-mcc-f1`
library used in this application.

The work of D.C. is funded by the European Union – Next Generation EU
programme, in the context of the National Recovery and Resilience Plan,
Investment Partenariato Esteso PE8 “Conseguenze e sfide dell’invecchiamento”,
Project Age-It (Ageing Well in an Ageing Society) and is partially supported
by Ministero dell’Università e della Ricerca of Italy under the “Dipartimenti
di Eccellenza 2023-2027” ReGAInS grant assigned to Dipartimento di Informatica
Sistemistica e Comunicazione at Università di Milano-Bicocca. The funders had
no role in study design, data collection and analysis, decision to publish, or
preparation of the manuscript.

# Conflict of Interest

The authors declare no conflict of interest in this work.

At the time of writing DHB is employed by an artificial intelligence company.
The software described in this paper was developed independently by DHB as a
personal project. His employer is not affiliated with the development of the
software, has no ownership rights, and did not provide any resources or
support for this work. The author's contributions to the project were made
entirely outside of their employment with any entity.

# Author Contributions

DHB: conceptualization, software implementation, application design, writing

DC: application design, domain expertise, scholarly integration, writing

# References
