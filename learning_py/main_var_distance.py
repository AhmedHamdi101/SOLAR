import util
from optimizer import Optimizer
import itertools
import time
import random
import numpy as np
from decision_maker import MethodSelector

def split_datasets_and_generate_joins(datasets, debug, seed, percentage):
    print("\n*** calling split_datasets_and_generate_joins in main_threshold.py ***")
    random.seed(seed)
    np.random.seed(seed)


    num_seen = int(len(datasets) * percentage)
    datasets_seen = random.sample(datasets, num_seen)
    datasets_unseen = [d for d in datasets if d not in datasets_seen]

    train_iter = max(len(datasets_seen), 5)
    test_iter = 40 - train_iter


    join_seen = []
    remaining_seen = set(datasets_seen)

    while remaining_seen:
        left = remaining_seen.pop()
        right = random.choice([d for d in datasets_seen if d != left])
        join_seen.append((left, right))

    while len(join_seen) < train_iter:
        pair = tuple(random.sample(datasets_seen, 2))
        join_seen.append(pair)

    join_unseen = [tuple(random.sample(datasets_unseen, 2)) for _ in range(test_iter)]


    if debug:
        print("---- datasets_seen size", len(datasets_seen))
        print("    ---- the following is datasets_seen")
        for ds in datasets_seen:
            print("        ----", ds)
        print("---- datasets_unseen size", len(datasets_unseen))
        print("    ---- the following is datasets_unseen")
        for ds in datasets_unseen:
            print("        ----", ds)
        print("---- join_seen size", len(join_seen))
        print("    ---- the following are all joins from join_seen")
        for jn in join_seen:
            print("        ----", jn)
        print("---- join_unseen size", len(join_unseen))
        print("    ---- the following are all joins from join_unseen")
        for jn in join_unseen:
            print("        ----", jn)
    #exit(0)
    return datasets_seen, datasets_unseen, join_seen, join_unseen



def run_scenario(dataset_dirs, distances, grid, selector, percentage, debug, experiment_mode, seed=2024):
    join_optimizer = Optimizer(seed)
    datasets = []
    for ddir in dataset_dirs:
        datasets += util.get_dataset_paths(ddir)
    
    datasets_seen, datasets_unseen, joins_seen, joins_unseen = split_datasets_and_generate_joins(datasets, debug,  seed, percentage)
    join_optimizer.offline_partitioning(joins_seen, distances[1], [grid], debug)
    util.compute_all_polygon_features(datasets)
    join_optimizer.training(datasets_seen, grid, debug)

    for distance in distances:
        print("the current distance is ", distance)
        print("\n\n *** evaluating seen cases *** ")
        join_optimizer.evaluate_model(joins_seen, distance, baselines=["GEOSPARK_QUADTREE", "GEOSPARK_KDBTREE"], selector = selector, debug=debug, experiment_mode=experiment_mode)
        print("\n\n *** evaluating unseen cases ***" )
        join_optimizer.evaluate_model(joins_unseen, distance, baselines=["GEOSPARK_QUADTREE", "GEOSPARK_KDBTREE"], selector = selector , debug=debug, experiment_mode=experiment_mode)


if __name__ == "__main__":
    default_grid = 8192
    debug = True
    experiment_mode = False
    seed = 2025316
    distances = [200,400,600,800]
    percentage = 0.8

    selector = MethodSelector()
    selector.load_model("new_pycode/my_method_selector.pkl")



    run_scenario(["/dataset_china","/dataset_us","/dataset_world","/dataset_other"], distances,  default_grid, selector, percentage, debug, experiment_mode, seed)
