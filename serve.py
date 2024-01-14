import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import Spacer, column, row
from bokeh.models import CheckboxGroup, LinearColorMapper, Paragraph, Slider
from bokeh.palettes import Colorblind8, Purples
from bokeh.plotting import figure
from bokeh.transform import transform

from distributions import NormalDistData
from metrics import Metrics


def threshold_slider_range_handler(attr, old, new):
    """Update threshold slider range based on total spread of data."""
    new_min = metrics.roc_thresholds.data["thresholds"].min()
    new_max = metrics.roc_thresholds.data["thresholds"].max()
    threshold_slider.start = new_min
    threshold_slider.end = new_max


def checkbox_callback(attr, old, new):
    """Update plot visibility based on checkbox status."""
    plot_roc.visible = 0 in checks1.active
    auc_bar.visible = 1 in checks1.active
    plot_pr.visible = 2 in checks1.active

    plot_cm.visible = 0 in checks2.active
    plot_mcc_f1.visible = 1 in checks2.active
    acc_bar.visible = 2 in checks2.active

    recall_bar.visible = 0 in checks3.active
    precision_bar.visible = 1 in checks3.active
    f1_bar.visible = 2 in checks3.active
    mcc_bar.visible = 3 in checks3.active


# Initial distribution settings
DEFAULT_N = 100
DEFAULT_MEAN_0 = 20.0
DEFAULT_MEAN_1 = 22.0
DEFAULT_SD = 3.0
DEFAULT_SKEW = 0.0
dist0 = NormalDistData(DEFAULT_N, DEFAULT_MEAN_0, DEFAULT_SD, DEFAULT_SKEW)
dist1 = NormalDistData(DEFAULT_N, DEFAULT_MEAN_1, DEFAULT_SD, DEFAULT_SKEW)

# Calculate all classification metrics
metrics = Metrics(dist0, dist1)


# Set up colorblind friendly colormap & shift order
cmap = list(Colorblind8)
cmap.insert(0, cmap.pop())


# Interactive GUI Sliders
slider_n0 = Slider(
    title="N",
    start=50,
    end=5000,
    step=10,
    value=DEFAULT_N,
    max_width=125,
    bar_color=cmap[0],
)
slider_mean0 = Slider(
    title="Mean",
    start=0,
    end=50,
    step=0.5,
    value=DEFAULT_MEAN_0,
    max_width=125,
    bar_color=cmap[0],
)
slider_sd0 = Slider(
    title="SD",
    start=0.1,
    end=20,
    step=0.1,
    value=DEFAULT_SD,
    max_width=125,
    bar_color=cmap[0],
)
slider_skew0 = Slider(
    title="Skew",
    start=-50,
    end=50,
    step=1,
    value=DEFAULT_SKEW,
    max_width=75,
    bar_color=cmap[0],
)
slider_n1 = Slider(
    title="N",
    start=50,
    end=5000,
    step=10,
    value=DEFAULT_N,
    max_width=125,
    bar_color=cmap[2],
)
slider_mean1 = Slider(
    title="Mean",
    start=0,
    end=50,
    step=0.5,
    value=DEFAULT_MEAN_1,
    max_width=125,
    bar_color=cmap[2],
)
slider_sd1 = Slider(
    title="SD",
    start=0.1,
    end=20,
    step=0.1,
    value=DEFAULT_SD,
    max_width=125,
    bar_color=cmap[2],
)
slider_skew1 = Slider(
    title="Skew",
    start=-50,
    end=50,
    step=1,
    value=DEFAULT_SKEW,
    max_width=75,
    bar_color=cmap[2],
)
threshold_slider = Slider(
    start=metrics.roc_thresholds.data["thresholds"].min(),
    end=metrics.roc_thresholds.data["thresholds"].max(),
    value=metrics.roc_thresholds.data["thresholds"].min(),
    step=0.001,
    title="classification threshold",
    max_width=125,
    bar_color=cmap[4],
    margin=(5, 5, 5, 50),
)


# Interactivity callback handling between plots & underlying data
slider_n0.on_change(
    "value",
    dist0.n_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
)
slider_mean0.on_change(
    "value",
    dist0.mean_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)
slider_sd0.on_change(
    "value",
    dist0.sd_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)
slider_skew0.on_change(
    "value",
    dist0.skew_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)
slider_n1.on_change(
    "value",
    dist1.n_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
)
slider_mean1.on_change(
    "value",
    dist1.mean_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)
slider_sd1.on_change(
    "value",
    dist1.sd_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)
slider_skew1.on_change(
    "value",
    dist1.skew_handler,
    metrics.threshold_line_y_handler,
    metrics.roc_curve_handler,
    metrics.pr_curve_handler,
    metrics.cm_handler,
    metrics.mcc_f1_curve_handler,
    metrics.metrics_handler,
    threshold_slider_range_handler,
)

threshold_slider.on_change(
    "value",
    metrics.threshold_line_x_handler,
    metrics.roc_threshold_dot_handler,
    metrics.pr_threshold_dot_handler,
    metrics.cm_handler,
    metrics.mcc_f1_threshold_dot_handler,
    metrics.metrics_handler,
)


# Checkboxes for toggling individual plots
PLOT_CHECKS1 = ["ROC Curve", "AUC", "PR Curve"]
PLOT_CHECKS2 = ["Confusion Matrix", "MCC-F1 Curve", "Accuracy"]
PLOT_CHECKS3 = ["Recall", "Precision", "F1", "MCC*"]

checks1 = CheckboxGroup(
    labels=PLOT_CHECKS1, active=[0, 1], margin=(-45, 5, 5, 70)
)  # top right bottom left
checks2 = CheckboxGroup(labels=PLOT_CHECKS2, active=[0], margin=(-45, 5, 5, -190))
checks3 = CheckboxGroup(labels=PLOT_CHECKS3, active=[], margin=(-45, 5, 5, -160))

checks1.on_change("active", checkbox_callback)
checks2.on_change("active", checkbox_callback)
checks3.on_change("active", checkbox_callback)


# ====================================================================
# PLOTS

# Distributions
plot_distributions = figure(
    title="Class Distributions",
    x_axis_label="Model Prediction (arbitrary)",
    y_axis_label="Count",
    plot_height=300,
    plot_width=325,
    toolbar_location=None
    # output_backend='webgl'
)
plot_distributions.line(
    "x", "y", source=dist0.kde_curve, line_color=cmap[0], line_width=2
)
plot_distributions.line(
    "x", "y", source=dist1.kde_curve, line_color=cmap[2], line_width=2
)
plot_distributions.line(
    "x", "y", source=metrics.threshold_line, line_color=cmap[4], line_width=4
)


# ROC Curve
plot_roc = figure(
    title="ROC Curve",
    x_axis_label="False Positive Rate",
    y_axis_label="True Positive Rate (Recall)",
    plot_height=300,
    plot_width=325,
    toolbar_location=None
    # output_backend='webgl'
)
plot_roc.line(
    "x", "y_upper", source=metrics.roc_curve, line_width=2, line_color=cmap[1]
)
plot_roc.line([0, 1], [0, 1], line_width=1, line_color="grey", line_dash="dashed")
plot_roc.scatter(
    "x",
    "y",
    source=metrics.roc_threshold_dot,
    size=13,
    fill_color=cmap[4],
    line_color=cmap[4],
)
# Add shading under ROC curve
plot_roc.varea(
    source=metrics.roc_curve,
    x="x",
    y1="y_lower",
    y2="y_upper",
    fill_color=cmap[1],
    alpha=0.1,
)


# ROC AUC Bar
auc_bar = figure(
    title="AUC",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
auc_bar.vbar(x=0.5, top="auc", source=metrics.metrics, width=0.5, fill_color=cmap[1])
auc_bar.y_range.start = 0.0
auc_bar.y_range.end = 1.0
auc_bar.xgrid.grid_line_color = None
auc_bar.xaxis.major_label_text_font_size = "0pt"
auc_bar.xaxis.major_tick_line_color = None
auc_bar.xaxis.minor_tick_line_color = None
auc_bar.line([0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed")


# PR Curve
plot_pr = figure(
    title="PR Curve",
    x_axis_label="True Positive Rate (Recall)",
    y_axis_label="Precision",
    y_range=[0.0, 1.04],
    plot_height=300,
    plot_width=325,
    toolbar_location=None,
    # output_backend="webgl",
)
plot_pr.line("x", "y_upper", source=metrics.pr_curve, line_width=2, line_color=cmap[1])
plot_pr.line(
    "x",
    "y_lower",
    source=metrics.pr_curve,
    line_width=1,
    line_color="grey",
    line_dash="dashed",
)
plot_pr.scatter(
    "x",
    "y",
    source=metrics.pr_threshold_dot,
    size=13,
    fill_color=cmap[4],
    line_color=cmap[4],
)
# Add shading for area under PR curve
plot_pr.varea(
    source=metrics.pr_curve,
    x="x",
    y1="y_lower",
    y2="y_upper",
    fill_color=cmap[1],
    alpha=0.1,
)


# Confusion Matrix
plot_cm = figure(
    title="Confusion Matrix",
    x_axis_label="Predicted",
    y_axis_label="True",
    x_range=[-0.5, 1.5],
    y_range=[1.5, -0.5],
    plot_height=300,
    plot_width=325,
    toolbar_location=None
    # output_backend='webgl'
)
cm_cmap = list(reversed(Purples[256]))[256 // 5 : -256 // 4]
mapper = LinearColorMapper(palette=cm_cmap)
plot_cm.rect(
    x="x",
    y="y",
    width=1,
    height=1,
    source=metrics.cm,
    line_color=None,
    fill_color=transform("cm_values", mapper),
)
plot_cm.axis.minor_tick_line_color = None
plot_cm.xaxis.ticker = [0, 1]
plot_cm.yaxis.ticker = [0, 1]
plot_cm.text(
    x="value_coord_x",
    y="value_coord_y",
    source=metrics.cm,
    text="cm_values",
    color="black",
    text_align="center",
    text_font_size={"value": "12px"},
)


# MCC-F1 Curve
# (MCC is unit normalized 0-1 instead of -1 to +1)
# Cao et al. 2020
plot_mcc_f1 = figure(
    title="MCC-F1",
    x_axis_label="F1-Score",
    y_axis_label="Unit Normalized MCC",
    y_range=[0, 1],
    plot_height=300,
    plot_width=325,
    toolbar_location=None
    # output_backend='webgl'
)
plot_mcc_f1.line("x", "y", source=metrics.mcc_f1_pts, line_width=2, line_color=cmap[1])
plot_mcc_f1.line(
    [0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed"
)
plot_mcc_f1.scatter(
    "x", "y", source=metrics.mcc_f1_dot, size=13, fill_color=cmap[4], line_color=cmap[4]
)


# Accuracy Bar
acc_bar = figure(
    title="Accuracy",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
acc_bar.vbar(
    x=0.5, top="accuracy", source=metrics.metrics, width=0.5, fill_color=cmap[1]
)
acc_bar.y_range.start = 0.0
acc_bar.y_range.end = 1.0
acc_bar.xgrid.grid_line_color = None
acc_bar.xaxis.major_label_text_font_size = "0pt"
acc_bar.xaxis.major_tick_line_color = None
acc_bar.xaxis.minor_tick_line_color = None
acc_bar.line([0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed")


# Recall Bar
recall_bar = figure(
    title="Recall",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
recall_bar.vbar(
    x=0.5, top="recall", source=metrics.metrics, width=0.5, fill_color=cmap[1]
)
recall_bar.y_range.start = 0.0
recall_bar.y_range.end = 1.0
recall_bar.xgrid.grid_line_color = None
recall_bar.xaxis.major_label_text_font_size = "0pt"
recall_bar.xaxis.major_tick_line_color = None
recall_bar.xaxis.minor_tick_line_color = None
recall_bar.line([0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed")


# Precision Bar
precision_bar = figure(
    title="Precision",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
precision_bar.vbar(
    x=0.5, top="precision", source=metrics.metrics, width=0.5, fill_color=cmap[1]
)
precision_bar.y_range.start = 0.0
precision_bar.y_range.end = 1.0
precision_bar.xgrid.grid_line_color = None
precision_bar.xaxis.major_label_text_font_size = "0pt"
precision_bar.xaxis.major_tick_line_color = None
precision_bar.xaxis.minor_tick_line_color = None
precision_bar.line(
    [0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed"
)


# F1-Score Bar
f1_bar = figure(
    title="F1-Score",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
f1_bar.vbar(
    x=0.5,
    top="f1",
    source=metrics.metrics,
    width=0.5,
    fill_color=cmap[1],
    line_color=cmap[1],
)
f1_bar.y_range.start = 0.0
f1_bar.y_range.end = 1.0
f1_bar.xgrid.grid_line_color = None
f1_bar.xaxis.major_label_text_font_size = "0pt"
f1_bar.xaxis.major_tick_line_color = None
f1_bar.xaxis.minor_tick_line_color = None


# Matthew's Correlation Coefficient Bar
mcc_bar = figure(
    title="MCC*",
    x_range=[0, 1],
    plot_height=300,
    plot_width=92,
    toolbar_location=None,
    # output_backend="webgl",
)
mcc_bar.vbar(
    x=0.5,
    top="mcc_norm",
    source=metrics.metrics,
    width=0.5,
    fill_color=cmap[1],
    line_color=cmap[1],
)
mcc_bar.y_range.start = 0.0
mcc_bar.y_range.end = 1.0
mcc_bar.xgrid.grid_line_color = None
mcc_bar.xaxis.major_label_text_font_size = "0pt"
mcc_bar.xaxis.major_tick_line_color = None
mcc_bar.xaxis.minor_tick_line_color = None
mcc_bar.line([0, 1], [0.5, 0.5], line_width=1, line_color="grey", line_dash="dashed")


# Initialize plot visibility
plot_roc.visible = 0 in checks1.active
auc_bar.visible = 1 in checks1.active
plot_pr.visible = 2 in checks1.active

plot_cm.visible = 0 in checks2.active
plot_mcc_f1.visible = 1 in checks2.active
acc_bar.visible = 2 in checks2.active

recall_bar.visible = 0 in checks3.active
precision_bar.visible = 1 in checks3.active
f1_bar.visible = 2 in checks3.active
mcc_bar.visible = 3 in checks3.active


# Add text blurb
# "Cao et al. 2020 https://arxiv.org/abs/2006.11278"
blurb = "*MCC is Unit Normalized MCC as in Cao et al. 2020"
txt = Paragraph(
    text=blurb,
    style=dict({"font-style": "italic", "font-size": "12px"}),
    width=200,
    height=200,
    margin=(265, 5, 5, 25),
)  # top right bottom left


# ====================================================================
# Arrange plots and widgets in a layout
spacer = Spacer(width=200, height=1)
slider_row1 = row(slider_n0, slider_mean0, slider_sd0, slider_skew0, spacer)
slider_row2 = row(
    slider_n1,
    slider_mean1,
    slider_sd1,
    slider_skew1,
    threshold_slider,
    checks1,
    checks2,
    checks3,
)
graph_row1 = row(plot_distributions, plot_roc, auc_bar, plot_pr)
graph_row2 = row(
    plot_cm, plot_mcc_f1, acc_bar, recall_bar, precision_bar, f1_bar, mcc_bar, txt
)
layout = column(slider_row1, slider_row2, graph_row1, graph_row2)


# Display
curdoc().add_root(layout)
