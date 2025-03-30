import util
from optimizer import Optimizer
import itertools
import time

if __name__ == "__main__":

    join_optimizer = Optimizer(2024)
    datasets = util.get_dataset_paths(xxx.default_data_path)
    
    datasets_seen, datasets_unseen, joins_seen, joins_unseen = join_optimizer.split_datasets_and_generate_joins(datasets, 0.8, True)
    
    import joinutil
    joinutil.run_spark_java()
    

    distances = [1000]
    default_grid = 8192
    default_sim_threshold = 0.5
    join_optimizer.training(datasets_seen, default_grid , False)

   

    for dist in distances:
        print("the current distance is " , dist) 
        join_unseen_reordered, max_sims, runtimes_unseen_ours = join_optimizer.evaluate_model(joins_unseen, dist , ["GEOSPARK_QUADTREE", "GEOSPARK_KDBTREE", "GEOSPARK_GRID"], default_sim_threshold, True)
        join_optimizer.evaluate_model(joins_seen, dist , ["GEOSPARK_QUADTREE", "GEOSPARK_KDBTREE"], default_sim_threshold, True)
        

    

