"""The analysis half of :class:`~snd.pipeline.cleaner.NoiseCleaner`.

Split in two so the metric definitions are readable without scrolling past the
matplotlib code: :class:`~snd.evaluation.cleaner_metrics.CleanerMetricsMixin`
computes, :class:`~snd.evaluation.cleaner_plots.CleanerPlotsMixin` draws.
``NoiseCleaner`` inherits the composition below, so ``cleaner.analyze()``,
``cleaner.calculate_relabeling_score(...)`` and every ``cleaner.plot_*`` call
still resolve on the cleaner instance.
"""
from snd.evaluation.cleaner_metrics import CleanerMetricsMixin
from snd.evaluation.cleaner_plots import CleanerPlotsMixin


class CleanerReportingMixin(CleanerMetricsMixin, CleanerPlotsMixin):
    """Ground-truth analysis, scoring and plotting for the noise cleaner."""
