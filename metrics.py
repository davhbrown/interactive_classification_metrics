import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from mcc_f1 import mcc_f1_curve
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from distributions import NormalDistData


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Metrics:
    """Calculates initial and updated binary classification metrics based on 2 input distributions."""

    def __init__(self, dist0: NormalDistData, dist1: NormalDistData):
        # All interactive & inter-related, so pass the instances and keep as ColumnDataSources as much as possible
        self.dist0 = dist0
        self.dist1 = dist1
        self.max_dist_y_value = self._get_dist_plot_max()

        # "True" binary labels
        # assigns 0s to the first distribution, 1s to the 2nd distribution
        self.y_true = ColumnDataSource(
            data=dict(
                data=np.concatenate(
                    (
                        np.zeros_like(self.dist0.raw_data.data["data"]),
                        np.ones_like(self.dist1.raw_data.data["data"]),
                    )
                )
            )
        )

        # Actual values from distribution
        # to compare against self.y_true for constructing an ROC curve or other
        self.y_score = ColumnDataSource(
            data=dict(
                data=np.concatenate(
                    (self.dist0.raw_data.data["data"], self.dist1.raw_data.data["data"])
                )
            )
        )

        # Make ROC Curve data
        (
            self.roc_curve,
            self.roc_thresholds,
            self.roc_threshold_dot,
        ) = self._make_roc_curve()

        # Make PR Curve data
        self.pr_curve, self.pr_thresholds, self.pr_threshold_dot = self._make_pr_curve()

        # Threshold line graphical object construction
        self.threshold_line = ColumnDataSource(
            data=dict(
                x=[
                    self.roc_thresholds.data["thresholds"].min(),
                    self.roc_thresholds.data["thresholds"].min(),
                ],
                y=[0.0, self._get_dist_plot_max(),],
            )
        )

        # MCC-F1 Curve
        (
            self.mcc_f1_pts,
            self.mcc_f1_thresholds,
            self.mcc_f1_dot,
        ) = self._make_mcc_f1_curve()

        # Make confusion matrix
        self.cm = self._get_confusion_matrix()

        # Metrics like AUC, recall, f1, etc.
        self.metrics = self._get_metrics()

    def _get_dist_plot_max(self):
        max0 = self.dist0.kde_curve.data["y"].max()
        max1 = self.dist1.kde_curve.data["y"].max()
        max_dist_y_value = np.maximum(max0, max1)
        return max_dist_y_value

    def _update_y(self):
        """Update predicted and true lables (`y` in the machine learning context). Not to be confused with (x,y) axes in plots."""
        y_true = dict(
            data=np.concatenate(
                (
                    np.zeros_like(self.dist0.raw_data.data["data"]),
                    np.ones_like(self.dist1.raw_data.data["data"]),
                )
            )
        )
        y_score = dict(
            data=np.concatenate(
                (self.dist0.raw_data.data["data"], self.dist1.raw_data.data["data"])
            )
        )
        self.y_true.data = y_true
        self.y_score.data = y_score
        return self.y_true, self.y_score

    def _make_roc_curve(self):
        # magic number, change to threshold_slider.value later if dot_handler inadequate
        INITIAL_THRESHOLD_VALUE = 0.0
        # Calculate ROC curve coordinates
        fpr, tpr, thresh = roc_curve(
            self.y_true.data["data"], self.y_score.data["data"]
        )
        # ROC CDS includes data for shading area under curve: y_lower & upper
        curve = ColumnDataSource(
            data=dict(x=fpr, y_upper=tpr, y_lower=np.zeros_like(tpr))
        )
        thresholds = ColumnDataSource(data=dict(thresholds=thresh))
        # Find nearest ROC point based on threshold
        roc_curve_idx = find_nearest_idx(
            thresholds.data["thresholds"], INITIAL_THRESHOLD_VALUE
        )
        dot = ColumnDataSource(
            data=dict(x=[fpr[roc_curve_idx]], y=[tpr[roc_curve_idx]])
        )
        return curve, thresholds, dot

    def _make_pr_curve(self):
        INITIAL_THRESHOLD_VALUE = 0.0

        # Calculate PR curve coordinates
        prec, recall, thresh = precision_recall_curve(
            self.y_true.data["data"], self.y_score.data["data"]
        )

        # PR-curve has a floor dependent on the class balance
        num_negative = self.dist0._n
        num_positive = self.dist1._n
        pr_baseline = num_positive / (num_positive + num_negative)
        pr_baseline = np.zeros_like(recall) + pr_baseline

        # PR-curve CDS includes data for shading area under curve
        curve = ColumnDataSource(data=dict(x=recall, y_upper=prec, y_lower=pr_baseline))
        thresholds = ColumnDataSource(data=dict(thresholds=thresh))

        # Find nearest PR point based on threshold
        pr_curve_idx = find_nearest_idx(
            thresholds.data["thresholds"], INITIAL_THRESHOLD_VALUE
        )
        dot = ColumnDataSource(
            data=dict(x=[recall[pr_curve_idx]], y=[prec[pr_curve_idx]])
        )
        return curve, thresholds, dot

    def _get_confusion_matrix(self):
        y_true = self.y_true.data["data"]
        y_score = self.y_score.data["data"]
        # Calculate predictions based on threshold
        thresh = self.threshold_line.data["x"][0]
        y_pred = (y_score >= thresh).astype("int")
        # Create confusion matrix data structure for plotting
        cm = confusion_matrix(y_true, y_pred)  # result: [[tn, fp], [fn, tp]]
        df = pd.DataFrame(
            {
                "x": [0, 1, 0, 1],
                "y": [1, 1, 0, 0],
                "cm_values": np.flip(cm, axis=0).flatten(),  # result: [fn, tp, tn, fp]
                "value_coord_x": [0, 1, 0, 1],
                "value_coord_y": [1, 1, 0, 0],
            }
        )
        cm = ColumnDataSource(df)
        return cm

    def _make_mcc_f1_curve(self):
        # Get predictions and MCC-F1 curve points
        mcc, f1, thresh = mcc_f1_curve(
            self.y_true.data["data"], self.y_score.data["data"]
        )
        curve = ColumnDataSource(data=dict(x=f1, y=mcc))
        thresholds = ColumnDataSource(data=dict(thresholds=thresh))
        # Find nearest MCC-F1 point based on threshold
        INITIAL_THRESHOLD_VALUE = 0.0
        mcc_f1_curve_idx = find_nearest_idx(
            thresholds.data["thresholds"], INITIAL_THRESHOLD_VALUE
        )
        dot = ColumnDataSource(
            data=dict(x=[f1[mcc_f1_curve_idx]], y=[mcc[mcc_f1_curve_idx]])
        )
        return curve, thresholds, dot

    def _get_metrics(self):
        # Convert for readability within this function
        y_true = self.y_true.data["data"].astype("int")
        y_score = self.y_score.data["data"]

        thresh = self.threshold_line.data["x"][0]
        y_pred = (y_score >= thresh).astype("int")

        cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)

        # Get metrics
        roc_auc = roc_auc_score(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = cr["0"]["recall"]
        precision = precision_score(y_true, y_pred, zero_division=1)
        npv = cr["0"]["precision"]
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_norm = (mcc + 1) / 2

        # Package
        metrics = ColumnDataSource(data=dict(roc_auc=[roc_auc]))
        metrics.data["avg_prec"] = [avg_prec]
        metrics.data["accuracy"] = [accuracy]
        metrics.data["recall"] = [recall]
        metrics.data["specificity"] = [specificity]
        metrics.data["precision"] = [precision]
        metrics.data["npv"] = [npv]
        metrics.data["f1"] = [f1]
        metrics.data["mcc_norm"] = [mcc_norm]
        return metrics

    # Threshold callback handlers (green line & dot)
    def threshold_line_x_handler(self, attr, old, new):
        """Controls horizontal movement when threshold changes."""
        self.threshold_line.data = dict(x=[new, new], y=[0.0, self.max_dist_y_value])

    def threshold_line_y_handler(self, attr, old, new):
        """Controls height changes when distributions change (dist.n or dist.sd)."""
        self.max_dist_y_value = self._get_dist_plot_max()
        x = self.threshold_line.data["x"][0]
        self.threshold_line.data = dict(x=[x, x], y=[0.0, self.max_dist_y_value])

    def roc_threshold_dot_handler(self, attr, old, new):
        """Update coordinates of roc_threshold dot."""
        roc_curve_idx = find_nearest_idx(self.roc_thresholds.data["thresholds"], new)
        fpr = self.roc_curve.data["x"]
        tpr = self.roc_curve.data["y_upper"]
        x, y = fpr[roc_curve_idx], tpr[roc_curve_idx]
        self.roc_threshold_dot.data = dict(x=[x], y=[y])

    def roc_curve_handler(self, attr, old, new):
        """Update ROC curve when distributions change."""
        self._update_y()
        new_curve, new_thresholds, new_dot = self._make_roc_curve()
        self.roc_curve.data = dict(new_curve.data)
        self.roc_thresholds.data = dict(new_thresholds.data)
        self.roc_threshold_dot.data = dict(new_dot.data)

    def pr_threshold_dot_handler(self, attr, old, new):
        """Update coordinates of PR curve dot."""
        pr_curve_idx = find_nearest_idx(self.pr_thresholds.data["thresholds"], new)
        recall = self.pr_curve.data["x"]
        prec = self.pr_curve.data["y_upper"]
        x, y = recall[pr_curve_idx], prec[pr_curve_idx]
        self.pr_threshold_dot.data = dict(x=[x], y=[y])

    def pr_curve_handler(self, attr, old, new):
        """Update PR curve when distributions change."""
        self._update_y()
        new_curve, new_thresholds, new_dot = self._make_pr_curve()
        self.pr_curve.data = dict(new_curve.data)
        self.pr_thresholds.data = dict(new_thresholds.data)
        self.pr_threshold_dot.data = dict(new_dot.data)

    def mcc_f1_curve_handler(self, attr, old, new):
        """Update MCC-F1 curve when distributions change."""
        self._update_y()
        new_points, new_thresholds, new_dot = self._make_mcc_f1_curve()
        self.mcc_f1_pts.data = dict(new_points.data)
        self.mcc_f1_thresholds.data = dict(new_thresholds.data)
        self.mcc_f1_dot.data = dict(new_dot.data)

    def mcc_f1_threshold_dot_handler(self, attr, old, new):
        """Update coordinates of threshold dot on MCC-F1 line."""
        mcc_f1_curve_idx = find_nearest_idx(
            self.mcc_f1_thresholds.data["thresholds"], new
        )
        f1 = self.mcc_f1_pts.data["x"]
        mcc = self.mcc_f1_pts.data["y"]
        x, y = f1[mcc_f1_curve_idx], mcc[mcc_f1_curve_idx]
        self.mcc_f1_dot.data = dict(x=[x], y=[y])

    def metrics_handler(self, attr, old, new):
        """Update metric values when distributions change."""
        self._update_y()
        metrics = self._get_metrics()
        self.metrics.data = dict(metrics.data)

    def cm_handler(self, attr, old, new):
        """Update confusion marix values."""
        self._update_y()
        cm = self._get_confusion_matrix()
        self.cm.data = dict(cm.data)
