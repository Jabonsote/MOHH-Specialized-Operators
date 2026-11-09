
from mohh.evaluation.params import Params
from mohh.evaluation.experiment import run_experiments
from mohh.evaluation.make_plots import make_figures

def execute_experiments(experiment_path, results_paths, operators, overrides: dict = None):

    evaluation_path = run_experiments(experiment_path, results_paths, operators, overrides)

    make_figures(evaluation_path)

    params = Params()
    params.reset_instance()

    from mohh.core.utilities.metrics import RunMetrics
    RunMetrics.reset_instance()

