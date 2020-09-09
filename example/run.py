import dacapo
import configargparse
import logging

parser = configargparse.ArgParser(
    default_config_files=['~/.config/dacapo', './dacapo.conf'])
parser.add(
    '-r', '--runs-config',
    is_config_file=True,
    help="The config file to use.")
parser.add(
    '-t', '--tasks',
    help="The tasks to run.")
parser.add(
    '-m', '--models',
    help="The models to use.")
parser.add(
    '-o', '--optimizers',
    help="The optimizers to use.")
parser.add(
    '-R', '--repetitions',
    help="How many repetitions to run.")
parser.add(
    '-v', '--validation-interval',
    help="How often to run validation.")
parser.add(
    '-s', '--snapshot-interval',
    help="How often to store a training batch.")
parser.add(
    '-b', '--keep-best-validation',
    help="If given, keep model checkpoint of best validation score with that "
         "name.")
parser.add(
    '-n', '--num-workers',
    help="Number of parallel processes to run.")

#logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    options = parser.parse_known_args()[0]

    print(options.tasks, options.models, options.optimizers)
    task_configs = dacapo.config.find_task_configs(eval(options.tasks))
    model_configs = dacapo.config.find_model_configs(eval(options.models))
    optimizer_configs = dacapo.config.find_optimizer_configs(
        eval(options.optimizers))

    print(task_configs, model_configs, optimizer_configs)

    configs = dacapo.enumerate_runs(        
        task_configs=task_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=eval(options.repetitions),
        validation_interval=eval(options.validation_interval),
        snapshot_interval=eval(options.snapshot_interval),
        keep_best_validation=eval(options.keep_best_validation))

    dacapo.run_all(configs, num_workers=int(options.num_workers))
