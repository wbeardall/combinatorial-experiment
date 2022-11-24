import json
import os
from pathlib import Path
import tempfile

import combinatorial_experiment
from combinatorial_experiment import CombinatorialExperiment, variables

def experiment_function(config):
    return {'called':json.dumps(config),'run_ok':True}

def test_run_experiment():
    argv = ['--base_config',os.path.join(Path(__file__).parent.absolute(),'configs','base_config.yml'),
            '--experiment_config',os.path.join(Path(__file__).parent.absolute(),'configs','experiment_config.yml'),
            '--output',os.path.join(tempfile.gettempdir(),'_comb_exp_output'),]
    records = CombinatorialExperiment.run_experiment(experiment_function,argv=argv)
    var = variables.deserialize_experiment_config(os.path.join(Path(__file__).parent.absolute(),'configs','experiment_config.yml'))
    assert len(records)==len(var)
    for el in records['METRIC_run_ok']:
        assert el
