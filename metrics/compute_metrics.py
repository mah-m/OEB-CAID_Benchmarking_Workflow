from __future__ import division
import io
import os
import pandas as pd
import json
from argparse import ArgumentParser
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Tuple, List, Union, Callable

from JSON_templates import JSON_templates
from parsers import parse_reference, parse_prediction, parse_thresholds

def ignore_numpy_warning(func: Callable) -> Callable:
    def wrapper(*arguments):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return func(*arguments)
    return wrapper

def main(args):
    input_participant = args.participant_data
    community = args.community_name
    challenges = args.challenges
    participant_name = args.participant_name
    gold_standards_dir = args.metrics_ref
    out_path = args.output

    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
            with open(out_path, mode="a"):
                pass
            print("built the path")
        except OSError as exc:
            print("OS error: {0}".format(exc) +
                  "\nCould not create output path: " + out_path)
    compute_metrics(input_participant, gold_standards_dir,
                    challenges, participant_name, community, out_path)


def binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate true and false positives per binary classification threshold.
    :param y_true: True labels of binary classifications
    :type y_true: np.ndarray
    :param y_score: Estimated probabilities or decision function
    :type y_score: np.ndarray
    :return:
        - fps: A count of false positives, at index i being the number of negative samples assigned a
        score >= thresholds[i]. The total number of negative samples is equal to fps[-1] (thus true negatives
        are given by fps[-1] - fps);
        - tps: An increasing count of true positives, at index i being the number of positive samples assigned a
        score >= thresholds[i]. The total number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps);
        - thresholds: Decreasing unique score values
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    pos_label = 1.0

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract the indices associated with the distinct values.
    # We also concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thr = y_score[threshold_idxs]

    return fps, tps, thr


def roc(fps: np.ndarray, tps: np.ndarray, thresholds: np.ndarray, drop_intermediates: bool = False) -> np.ndarray:
    """Compute Receiver operating characteristic (ROC).
    :param fps: decreasing count of false positives
    :type fps: np.ndarray
    :param tps: increasing count of true positives
    :type tps:  np.ndarray
    :param thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr. `thresholds[0]`
        represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`
    :type thresholds: np.ndarray
    :param drop_intermediates: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC
    curve. This is useful in order to create lighter  ROC curves.
    :type drop_intermediates: bool
    :return:
        - fpr: Increasing false positive rates such that element i is the false positive rate of predictions
        with score >= thresholds[i];
        - tpr: Increasing true positive rates such that element i is the true positive rate of predictions
        with score >= thresholds[i];
        - thresholds:  Decreasing thresholds on the decision function used to compute fpr and tpr. `thresholds[0]`
        represents no instances being predicted and is arbitrarily set to `max(thresholds) + 1`.
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """
    if drop_intermediates is True and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return np.array([fpr, tpr, thresholds], dtype=np.float64).round(3)


@ignore_numpy_warning
def pr(fps: np.ndarray, tps: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Compute precision-recall pairs for different probability thresholds
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    y axis.
    :param fps: decreasing count of false positives
    :type fps: np.ndarray
    :param tps: increasing count of true positives
    :type tps:  np.ndarray
    :param thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr. `thresholds[0]`
        represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`
    :type thresholds: np.ndarray
    :return:
        - precision : Increasing precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
        - recall : Increasing recall values such that element i is the recall of predictions with score >= thresholds[i]
        and the last element is 0.
        - thresholds : Decreasing thresholds on the decision function used to compute precision and recall.
    """

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    recall[np.isnan(recall)] = 0

    return np.array([np.r_[1, precision], np.r_[0, recall], np.r_[thresholds[0] + 1, thresholds]], dtype=np.float64)\
        .round(3)


def confmat(fps: np.ndarray, tps: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for different probability thresholds
    Confusion matrix `[[tn fp] [fn tp]]` for the binary case with labels [0,1] is computed for each threshold.
    Computation starts from the count of fp (`fps` param) and tp (`tps` param) for each threshold. For each
    threshold t in a series of decreasing threshold, $tn_t$ is calculated as $p - fp_t$ where $p$ is the number
    of positives labels and is represented by the last element of
    :param fps: decreasing count of false positives
    :type fps: np.ndarray
    :param tps: increasing count of true positives
    :type tps:  np.ndarray
    :return: array of shape (len(fps), 4) with the count of TNs FPs FNs adn TPs for each couple of values (FP, TP)
        in `fps`, `tps` params.
    """
    # true negatives are given by
    tns = fps[-1] - fps
    # false negatives are given by
    fns = tps[-1] - tps
    # tn, fp, fn, tp
    return np.array([tns, fps, fns, tps], dtype=np.float64)


# TODO: this cannot work since once in a dataframe ref and pred will always have the same length.
#  Missing labels will be replaced by nans and those nans will be indistiguishable from masked regions in reference
#  Nans in predictions should instead be safe to check
def find_length_mismatches(p: pd.DataFrame) -> List[str]:
    """Compare lengths of targets in reference and prediction
    Store and log on the warning level ids of targets with inconsistent lengths when found,
    then return a list of the ids with inconsistent lengths.
    :param p: aligned reference and prediction. It is expected in the format:
            |          |   | ref    | ref | predname | predname |
            |----------|---|--------|-----|----------|----------|
            |          |   | states | seq | states   | scores   |
            | target 1 | 1 | 1      | M   | 1        | 0.789    |
            | target 1 | 2 | 1      | S   | 0        | 0.456    |
    :type p: pd.DataFrame
    :return: list of target ids where the length of the reference is different from the length of the prediction
    """
    inconsistent_targets = []
    lbl = p.columns.get_level_values(0).unique()[1]
    for tgt, tgt_aligned in p.groupby(level=0):
        ps = tgt_aligned[(lbl, "states")].values
        rs = tgt_aligned[("ref", "states")].values

        # if np.any(np.isnan(ps)) and not np.all(np.isnan(ps)):
        if len(ps) < len(rs):
            inconsistent_targets.append(tgt)
        # if np.any(np.isnan(rs)):
        elif len(ps) > len(rs):
            inconsistent_targets.append(tgt)

    return inconsistent_targets


def align_reference_prediction(ref: dict, pred: dict, drop_missing: bool = True) -> Tuple[pd.DataFrame, List]:
    # merge reference a prediction dicts and cast to Pandas.DataFrame
    aln_pred = pd.DataFrame({**ref, **pred})
    predname = aln_pred.columns.get_level_values(0)[-1]

    # check for length mismatch between reference and prediction
    wrong_len_preds = find_length_mismatches(aln_pred)
    # remove targets with length mismatch
    aln_pred = aln_pred.loc[~aln_pred.index.get_level_values(0).isin(wrong_len_preds)]

    # remove rows with nan (now it's only possible if all residues are missing)
    isnan = aln_pred.isna().all()
    isnan = isnan[isnan == 1].index.tolist()
    if isnan:
        for p in isnan:
            aln_pred[p] = aln_pred[(p[0], 'states')]

    if drop_missing is True:
        aln_pred = aln_pred.dropna(0)

    return aln_pred, wrong_len_preds


def balanced_accuracy(nd_cmat: np.ndarray) -> np.ndarray:
    c = nd_cmat.T.reshape(nd_cmat.shape[1], 2, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diagonal(c, axis1=1, axis2=2) / c.sum(axis=2)
    score = np.nanmean(per_class, axis=1)
    return score


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: Union[float, int] = 1) -> np.ndarray:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    denom[denom == 0.] = 1  # avoid division by 0
    fbeta = ((1 + beta2) * precision * recall) / denom
    return fbeta


def negative_predictive_value(tn, fn):
    denom = tn + fn
    return np.divide(tn, denom, out=np.zeros_like(tn).astype(float), where=denom != 0)


def matt_cc(tn, fp, fn, tp):
    numer = (tp*tn - fp*fn)
    denom = (np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)))
    mcc = np.divide(numer, denom, out=np.zeros_like(numer).astype(float), where=denom != 0)
    return mcc


def auc(x, y):
    """ Compute Area Under the Curve (AUC) using the trapezoidal rule.
    :param x: x coordinates. These must be either monotonic increasing or monotonic decreasing
    :type x: np.ndarray
    :param y: y coordinates
    :type y: np.ndarray
    :return: area under the curve
    :rtype: float
    """

    if x.shape[0] < 2:
        area = np.nan
    else:
        direction = 1
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                # logging.error("direction of x argument in auc function is neither increasing nor decreasing.exiting")
                area = np.nan

        area = direction * np.trapz(y, x)
        if isinstance(area, np.memmap):
            # Reductions such as .sum used internally in np.trapz do not return a
            # scalar by default for numpy.memmap instances contrary to
            # regular numpy.ndarray instances.
            area = area.dtype.type(area)
    return area


def get_metrics(roc_curve, pr_curve, cmats: np.ndarray) -> dict:
    # TODO: check if it's really necessary to split.squeeze cmats
    # unpack per-threshold confusion matrices

    if cmats.shape == (4, 2):
        cmats = np.squeeze(np.split(cmats, 4, 0))
    tn, fp, fn, tp = cmats

    # remove first element (it's artificially added in pr func)
    ppv = pr_curve[0][1:]  # precision
    tpr = pr_curve[1][1:]  # sensitivity / recall

    # remove first element (they're calculated from an artificially added threshold in roc func)
    fpr = roc_curve[0][1:]  # fall-out
    tnr = 1 - fpr  # specificity / selectivity
    fnr = 1 - tpr  # miss-rate

    # compute other metrics
    bacc = balanced_accuracy(cmats)
    f1 = fbeta(ppv, tpr)
    f2 = fbeta(ppv, tpr, beta=2)
    f05 = fbeta(ppv, tpr, beta=.5)
    mcc = matt_cc(tn, fp, fn, tp)
    npv = negative_predictive_value(tn, fn)
    fom = 1 - npv  # false omission rate (for keyword is reserved)
    inf = tpr + tnr - 1  # bookmaker informedness
    mk = ppv + npv - 1  # markedness
    csi = tp / (tp + fn + fp)  # critical score index / threat score (doesn't need a func b/c denom can never be 0)

    return dict(npv=npv, ppv=ppv, tpr=tpr, tnr=tnr, fpr=fpr, fnr=fnr, fom=fom, csi=csi,
                bac=bacc, f1s=f1, f2s=f2, f05=f05, mcc=mcc, inf=inf, mk=mk)


def get_default_threshold(thresholds, predname, pred):
    default_thr = None

    if thresholds is not None:
        default_thr = thresholds.get(predname)

    if default_thr is None:
        default_thr = calculate_default_threshold(pred)

    return default_thr


def calculate_default_threshold(pred: np.ndarray) -> float:
    thr = pred[pred[:, 0] == 1][:, 1].min()
    return thr


def calc_curves_and_metrics(ytrue, yscore):
    fps, tps, thr = binary_clf_curve(ytrue, yscore)
    roc_curve = roc(fps, tps, thr)
    pr_curve = pr(fps, tps, thr)
    cmat = confmat(fps, tps)
    
    metrics = get_metrics(roc_curve, pr_curve, cmat)
    
    return roc_curve, pr_curve, cmat, metrics



def summary_metrics(roc_curve, pr_curve):
    ppv, tpr, _ = pr_curve
    auc_roc = auc(*roc_curve[:-1])
    auc_pr = auc(tpr, ppv)
    aps = -np.sum(np.diff(tpr[::-1]) * ppv[::-1][:-1])
    return dict(aucroc=np.round(auc_roc, 3), aucpr=np.round(auc_pr, 3), aps=np.round(aps, 3))


def dataset_curves_and_metrics(ytrue, yscore, predname):
    roc_curve, pr_curve, cmat, metrics = calc_curves_and_metrics(ytrue, yscore)
    smry_metrics = summary_metrics(roc_curve, pr_curve)
    
    indexes, values = zip(*metrics.items())
    metrics = pd.DataFrame(values,
                           columns=roc_curve[2][1:],
                           index=pd.MultiIndex.from_product([[predname], indexes])).round(3)

    roc_df = pd.DataFrame(roc_curve[:-1].T,
                          columns=pd.MultiIndex.from_product([[predname], [smry_metrics["aucroc"]], ["fpr", "tpr"]],
                                                             names=["predictor", "auc", "metric"]),
                          index=roc_curve[-1].round(3))

    pr_df = pd.DataFrame(pr_curve[1::-1].T,
                         columns=pd.MultiIndex.from_product(
                             [[predname], [smry_metrics["aucpr"]], [smry_metrics["aps"]], ["tpr", "ppv"]],
                             names=["predictor", "auc", "aps", "metric"]),
                         index=pr_curve[-1].round(3))

    cmat = pd.DataFrame(zip(*cmat),
                        columns=pd.MultiIndex.from_product([[predname], ["tn", "fp", "fn", "tp"]]),
                        index=roc_curve[-1][1:].round(3)).astype(int)

    return roc_df, pr_df, cmat, metrics, smry_metrics



def target_curves_and_metrics(aln_refpred, predname):

    target_metrics = {}
    for tgt, tgt_scores in aln_refpred.groupby(level=0):
        roc_tgt, pr_tgt, cmat_tgt, metrics_tgt = calc_curves_and_metrics(tgt_scores[('ref', 'states')].values,
                                                                         tgt_scores[(predname, 'scores')].values)
        # save in a data-structure easily convertible to pd.DataFrame
        tgt_d = {(tgt, m): dict(np.stack([roc_tgt[2][1:], metrics_tgt[m]], axis=1)) for m in metrics_tgt}
        # update metrics dict
        target_metrics = {**target_metrics, **tgt_d}

    target_metrics = pd.DataFrame(target_metrics).round(3)\
        .sort_index(ascending=False)\
        .fillna(method='ffill')\
        .fillna(method='backfill').T
    return target_metrics


def compute_metrics( prediction, gold_standards_dir, challenges,participant_name, community, outpath="."):
    ALL_ASSESSMENTS = []
    for challenge in challenges:
        ASSESSMENTS = []
        reference = os.path.join(gold_standards_dir, challenge+".txt")
        outpath = Path(outpath)
        # outpath.mkdir(parents=True, exist_ok=True)
        reference = Path(reference)
        ref_obj, accs = parse_reference(reference.resolve(strict=True))  # resolve raises an error if file doesn't exists
        provided_thr = None

        all_preds = {}
        thresholds = {}
               
        predname = participant_name # Path(prediction).stem # participant name
        
        pred_obj = parse_prediction(prediction, accs, predname)  # returns dict
        aln_ref_pred, wrong_tgt = align_reference_prediction(ref_obj, pred_obj)  # remove targets w/ errors     

        all_preds.update(pred_obj)  # add reference to be aligned with all preds

        roc_curve, pr_curve, cmat, dataset_metrics, smry_metrics = dataset_curves_and_metrics(
            aln_ref_pred[('ref', 'states')].values, aln_ref_pred[(predname, 'scores')].values,predname)
        
                  
        target_metrics = target_curves_and_metrics(aln_ref_pred, predname)    

        # {<label>: <threshold>} for each threshold a file will be saved with metrics optimized for that threshold
        thresholds = {"default": get_default_threshold(provided_thr, predname, aln_ref_pred[predname].to_numpy()),
                    **dataset_metrics.idxmax(1).loc[predname].to_dict()}
            # find metrics of current pred for each threshold in <thresholds>; store to be later joined with other preds
        
        default_threshold=thresholds["default"]
        for thr in thresholds:
            if thr == "default":
                default_dts_metrics=dict(dataset_metrics[default_threshold])
                for key,value in default_dts_metrics.items():
                    assessment=compute_metric_assessment(participant_name, community, challenge, f'dts_{key[1]}', value)
                    ASSESSMENTS.append(assessment)
            else:
                assessment=compute_metric_assessment(participant_name, community, challenge, f'opt_{thr}', thresholds[thr])
                ASSESSMENTS.append(assessment)
        
        for smry in smry_metrics:
            assessment=compute_metric_assessment(participant_name, community, challenge, f'{smry}', smry_metrics[smry])
            ASSESSMENTS.append(assessment)
        # print(dataset_metrics[default_threshold].unstack().assign(**smry_metrics,thr=default_threshold))
        for key,value in cmat.loc[default_threshold].items():
            assessment=compute_metric_assessment(participant_name, community, challenge, f'{key[1]}', value)
            ASSESSMENTS.append(assessment)
           
        # print(pd.concat([target_metrics[default_threshold].unstack()], keys=[predname]).assign(thr=default_threshold))
        target_df=pd.concat([target_metrics[default_threshold].unstack()], keys=[predname])
        target_mean=target_df.mean()
        for key,value in target_mean.items():
            assessment=compute_metric_assessment(participant_name, community, challenge, f'tgt_{key}_avg', round(value,3))
            ASSESSMENTS.append(assessment)
       
        ALL_ASSESSMENTS.extend(ASSESSMENTS)

    with io.open(outpath, mode='w', encoding="utf-8") as f:
        jdata = json.dumps(ALL_ASSESSMENTS, sort_keys=True, indent=4, separators=(',', ': '))
        f.write(jdata)

        
def compute_metric_assessment(participant_name, community, challenge, metric_name, metric):
    data_id = f'{community} : {challenge}_{metric_name}_{participant_name}_A'
    assessment = JSON_templates.write_assessment_dataset(data_id, community, challenge, participant_name, metric_name, metric)
    return assessment


if __name__ == '__main__':
    parser = ArgumentParser()
    # prediction
    parser.add_argument("-i", "--participant_data", help="list of the predictions", required=True)
    parser.add_argument("-com", "--community_name",
                        help="name of benchmarking community", required=True)
    # get the name of ref files by joining the name with ref dir
    parser.add_argument("-c", "--challenges", nargs='+',
                        help="list of challenges", required=True)
    parser.add_argument("-p", "--participant_name",
                        help="name of the tool used for the prediction", required=True)
    # reference directory
    parser.add_argument("-m", "--metrics_ref", help="dir that contains metrics reference datasets for all challenges", required=True)
    parser.add_argument("-o", "--output", help="output path where assessment JSON files will be written ", required=False)
    args = parser.parse_args()

    main(args)
