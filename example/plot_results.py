import dacapo

if __name__ == "__main__":

    print("Finding runs...")
    tasks = dacapo.config.find_task_configs('tasks/neuron_segmentation')
    models = dacapo.config.find_model_configs('models/3d')
    optimizers = dacapo.config.find_optimizer_configs('optimizers')
    runs = dacapo.enumerate_runs(tasks, models, optimizers, repetitions=5, validation_interval=0, snapshot_interval=0, keep_best_validation=None)

    print(tasks)
    print(models)
    print(optimizers)
    print(runs)

    print("Fetching data...")
    store = dacapo.store.MongoDbStore()
    for run in runs:
        store.sync_run(run)
        store.read_training_stats(run)
        store.read_validation_scores(run)

    print("Plotting...")
    dacapo.analyze.plot_runs(runs, smooth=100, validation_score='voi_sum')
    #again.analyze.plot_runs(runs, smooth=100)

    input("Enter to continue")

    import numpy as np

    def get_best(self, score_name=None, higher_is_better=True):

        names = self.get_score_names()

        best_scores = {name: [] for name in names}
        for iteration_scores in self.scores:
            ips = np.array([
                parameter_scores['scores']['average'][score_name]
                for parameter_scores in iteration_scores.values()
            ])
            print(ips[:10])
            print(np.isnan(ips[:10]))
            ips[np.isnan(ips)] = -np.inf if higher_is_better else np.inf
            print(ips[:10])
            i = np.argmax(ips) if higher_is_better else np.argmin(ips)
            for name in names:
                best_scores[name].append(
                    list(iteration_scores.values())[i]['scores']['average'][name]
                )
        return best_scores



    best = get_best(run.validation_scores, 'fscore') 
    print(best['detection_fscore'])
