from joinutil import run_join_query
from util import compute_histogram
import pickle
from util import bert_encoder, compute_jsd_distance, print_statistics
import time
import neuralnet
import itertools
import numpy as np
import os
import random
from collections import defaultdict
from itertools import combinations
import math
import torch
import json

class Optimizer():
    def __init__(self, seed):
        self.seed = seed
        self.dataset_encoding_repos = []
        self.encoding_path_mappings = {}

    def split_datasets_and_generate_joins(self, datasets, split_ratio, debug, mode): 
        print("\n*** callining split_datasets_and_generate_joins in optimizer.py ***")
        random.seed(self.seed)
        np.random.seed(self.seed)    
        
        if mode == "train_china_test_us":
            datasets_seen = [d for d in datasets if d.endswith("china.csv")]
            datasets_unseen = [d for d in datasets if d.endswith("us.csv")]
        
        elif mode == "train_us_test_china":
            datasets_seen = [d for d in datasets if d.endswith("us.csv")] 
            datasets_unseen = [d for d in datasets if d.endswith("china.csv")]
        
        elif mode == "train_us_china_test_world":
            datasets_seen = [d for d in datasets if d.endswith("us.csv")] + [d for d in datasets if d.endswith("china.csv")]
            datasets_unseen = [d for d in datasets if d.endswith("world.csv")]
            
        #elif mode == "train_world_test_uschina":
        #    datasets_seen = [d for d in datasets if d.endswith("world.csv")]
        #    datasets_unseen = [d for d in datasets if d.endswith("us.csv")] + [d for d in datasets if d.endswith("china.csv")]
        
        elif mode == "global":
            random.shuffle(datasets) 
            num_seen = int(len(datasets) * split_ratio)     
            datasets_seen = datasets[:num_seen]
            datasets_unseen = datasets[num_seen:] 
        
        
        if mode == "global":
            train_iter = 50
            test_iter = 20
            
        else:
            train_iter = 30
            test_iter = 15
       
        join_seen = []
        remaining_seen = set(datasets_seen)  # Keep track of datasets that need to appear in left position
        
        while remaining_seen:
            left = remaining_seen.pop()
            right = random.choice([d for d in datasets_seen if d != left])
            join_seen.append((left, right))
        
        # Fill up remaining join_seen pairs randomly
        while len(join_seen) < train_iter:
            pair = tuple(random.sample(datasets_seen, 2))
            join_seen.append(pair)
        
        join_unseen = [tuple(random.sample(datasets_unseen, 2)) for _ in range(test_iter)]
        
        if debug:
            print("the current train-test split mode is ", mode)
        
            print("---- datasets_seen size", len(datasets_seen))
            print("    ---- the following is datasets_seen ")
            for dataset in datasets_seen:
                print("        ---- ", dataset)
            
            print("---- datasets_unseen size", len(datasets_unseen))
            print("    ---- the following is datasets_unseen ")
            for dataset in datasets_unseen:
                print("        ---- ", dataset)

            print("---- join_seen size", len(join_seen))
            print("    ---- the following are all joins from join_seen")
            for join in join_seen:
                print("        ---- ", join)
                
            print("---- join_unseen size", len(join_unseen))
            print("    ---- the following are all joins from join_unseen")
            for join in join_unseen:
                print("        ---- ", join)

        return datasets_seen, datasets_unseen, join_seen, join_unseen




    def offline_partitioning(self, joins, dist, histogram_divs, debug):
        print("**** \n starting the offline partitioning phase ****")

        print("---- total number of joins to run in the cold start is ", len(joins), " ---- ")
        join_runtime_ours = []
        join_runtime_geospark_quadtree_original = []
        join_sizes = []
        for join in joins:
            if debug:
                print("\n    ---- current join " , join)
            
            # we need to run the actually join here in order to retrieve the partitioner
            join_para = {"joinDistance" : dist, "partitioner": "UNIVERSAL_QUADTREE_ONTHEFLY"}
            join_runtime, join_size = run_join_query(join[0], join[1], join_para)
            join_runtime_ours.append(join_runtime)

            join_para = {"joinDistance" : dist, "partitioner": "GEOSPARK_QUADTREE"}
            join_runtime_geospark, join_size_geospark = run_join_query(join[0], join[1], join_para)
            join_runtime_geospark_quadtree_original.append(join_runtime_geospark)

            join_sizes.append(join_size)

            if join_size != join_size_geospark:
                print("ERROR, size of two join results is different", " our size ", join_size , " baseline size " , join_size_geospark)
                exit(0)

            if debug:
                print("        ---- runtime ours", join_runtime, " runtime original geospark quadtree " , join_runtime_geospark ," join size", join_size)

            # After the running the join and storing the partitioner, compute the statistics (histogram) of the TWO joined datasets
            for div in histogram_divs:
                compute_histogram(join[0], div)
                compute_histogram(join[1], div)

        print("---- runtime for all ours ----")
        print(join_runtime_ours)
        print("---- runtime for all geospark original quadtree ----")
        print(join_runtime_geospark_quadtree_original)
        print("---- join sizes ----")
        print(join_sizes)
        print("----the following is the stat for our modified quadtree ----")
        print_statistics(join_runtime_ours)


    def training(self, datasets, div, debug):
        print("\n\n\n **** calling the training method in optimizer.py ****")
        print("---- total number of datasets is ", len(datasets))
        dataset_pairs = list(itertools.combinations(datasets, 2)) + [(dataset, dataset) for dataset in datasets]

        encodings_and_diff = []
        if debug:
            print("    ---- dataset has size", len(datasets))
        for pair in dataset_pairs:
            dataset1, dataset2 = pair

            hist1_path = f"stat/histogram/{div}/{dataset1.split('datasets/')[-1].replace('/', '@')}.pkl"
            hist2_path = f"stat/histogram/{div}/{dataset2.split('datasets/')[-1].replace('/', '@')}.pkl"

            with open(hist1_path, 'rb') as f1:
                hist1 = pickle.load(f1)

            with open(hist2_path, 'rb') as f2:
                hist2 = pickle.load(f2)

            dist = compute_jsd_distance(hist1, hist2)

            dataset1_encoding = bert_encoder(dataset1, debug)
            #print("    ---- dataset1 encoding is", dataset1_encoding)
            dataset2_encoding = bert_encoder(dataset2, debug)
            #print("    ---- dataset2 encoding is", dataset2_encoding)
            
            encoding_key1 = dataset1_encoding.tobytes()
            encoding_key2 = dataset2_encoding.tobytes()

            if encoding_key1 not in self.encoding_path_mappings:
                self.dataset_encoding_repos.append(dataset1_encoding)
                self.encoding_path_mappings[encoding_key1] = dataset1

            if encoding_key2 not in self.encoding_path_mappings:
                self.dataset_encoding_repos.append(dataset2_encoding)
                self.encoding_path_mappings[encoding_key2] = dataset2


            print(dataset1, " AND " , dataset2, "  their distance is ", dist)

            encodings_and_diff.append((dataset1_encoding, dataset2_encoding, dist))

            #if debug:
            #    print("    ---- the path to dataset1 is " , dataset1)
            #    print("    ---- the path to dataset2 is " , dataset2)
            #    print("    ---- the path to histogram1 is ", hist1_path)
            #    print("    ---- the path to histogram2 is ", hist2_path)
            #    print("    ----  the distance between this two data histogram is ", dist)
            #    print("    ---- the diff between two dataset (for debugging) is ", np.sum(dataset1_encoding - dataset2_encoding))
            #    print("    ------------------")

        
        start = time.time()
        self.trainer = neuralnet.SiameseModelTrainerFusion()
        self.model = self.trainer.train_model(encodings_and_diff)
        end = time.time()
        os.makedirs("./learning_py/trainer_meta_data", exist_ok=True)
        self.save_model("./learning_py/trainer_meta_data")
        print("---- training takes time ", (end - start), " current div is ",div)
        
        
    
    def save_model(self, path: str):
    

        model_path = os.path.join(path, "decider.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"[Optimizer] Saved model weights to {model_path}")

        # save both structures directly with pickle
        repo_path = os.path.join(path, "encodings_and_map.pkl")
        with open(repo_path, "wb") as f:
            pickle.dump({
                "dataset_encoding_repos": self.dataset_encoding_repos,
                "encoding_path_mappings": self.encoding_path_mappings,
                }, f)
        print(f"[Optimizer] Saved encodings + mappings to {repo_path}")



    def load_model(self, path: str):
        
        model_path = os.path.join(path, "decider.pt")
        self.model = neuralnet.SiameseNetworkFusion()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.trainer = neuralnet.SiameseModelTrainerFusion()
        self.trainer.model = self.model

        repo_path = os.path.join(path, "encodings_and_map.pkl")
        with open(repo_path, "rb") as f:
            data = pickle.load(f)
        self.dataset_encoding_repos = data["dataset_encoding_repos"]
        self.encoding_path_mappings = data["encoding_path_mappings"]

        print(f"[Optimizer] Loaded model + encodings/mappings from {repo_path}")

    
    # ["GEOSPARK_QUADTREE", "GEOSPARK_KDBTREE", "GEOSPARK_GRID"]
    def evaluate_model(self, joins, dist, baselines, selector, debug, experiment_mode = False):
        print("\n\n\n **** starting to evaluating the model, there are total of ", len(joins), " to evaluate ****")

        if experiment_mode:
            reuse_runtimes = []
            onthefly_runtimes = []
            max_sim_scores = []
            join_sizes = []
            
        else:
            runtimes = []
            baseline_runtimes = []
            match_times = [] 
            decision_times = []
            join_sizes = []
            decisions = []

        for join in joins:
            dataset1, dataset2 = join[0], join[1]
            dataset1_encoding = bert_encoder(dataset1, False)
            dataset2_encoding = bert_encoder(dataset2, False)

            
            if experiment_mode:
                reuse_runtime, onthefly_runtime, max_sim_score, join_size = self.run_join_using_model_againist_baseline(dataset1, dataset2,  dataset1_encoding, dataset2_encoding, dist, baselines, selector, debug, experiment_mode)
                reuse_runtimes.append(reuse_runtime)
                onthefly_runtimes.append(onthefly_runtime)
                max_sim_scores.append(max_sim_score)
                join_sizes.append(join_size)
            
            else:
                our_join_runtime, baseline_join_runtimes, match_time, decision_time, join_size, decision = self.run_join_using_model_againist_baseline(dataset1, dataset2,  dataset1_encoding, dataset2_encoding, dist, baselines, selector, debug, experiment_mode)
                runtimes.append(our_join_runtime)
                join_sizes.append(join_size)
                baseline_runtimes.append(baseline_join_runtimes)
                match_times.append(match_time)
                decision_times.append(decision_time)
                decisions.append(decision)
                
                
        if experiment_mode:
            print(" the following is the runtime of always reusing the partitioner ")
            print(reuse_runtimes)
            print(" the following is the runtime of on-the-fly partitioning ")
            print(onthefly_runtimes)
            print(" the following is the maximum similarity scores  ")
            print(max_sim_scores)
            print(" the following is the join sizes ")
            print(join_sizes)
            
        else:
            print(" the following is the runtime of runtimes")
            print(runtimes)
            print("---- the decision variables")
            print(decisions)
            print("---- match times" )
            print(match_times)
            print("---- decision times")
            print(decision_times)
            

            if baselines is not None:
                for i in range(len(baselines)):
                    print("    ----  baseline ", baselines[i] , " ----")
                    print([sublist[i] for sublist in baseline_runtimes])



        
    def run_join_using_model_againist_baseline(self, dataset1, dataset2, dataset1_encoding, dataset2_encoding, dist, baselines, selector, debug, experiment_mode):
        start_matching_time =  time.time()
        index1, pred_dist1 = self.trainer.find_most_similar(dataset1_encoding, self.dataset_encoding_repos)
        index2, pred_dist2 = self.trainer.find_most_similar(dataset2_encoding, self.dataset_encoding_repos)
        end_matching_time =  time.time()
        match_time = (end_matching_time - start_matching_time) * 1000
        
        min_dist = min(pred_dist1, pred_dist2)
        max_sim = 1 - min_dist
        join_seq = [dataset1, dataset2] if pred_dist1 < pred_dist2 else [dataset2, dataset1]
        our_join_runtime = -1
        baseline_join_runtimes = []
        if debug:
            print("\n---- current join ", dataset1, dataset2)
            print("    ----   pred_dist1 ", pred_dist1, " pred_dist2 ", pred_dist2, " min dist is ", min_dist, "max similarity is ", max_sim)
            print("    ---- the join sequence is ", join_seq)
        
        
        start_decision_time =  time.time()
        matched_dataset_encoding = self.dataset_encoding_repos[index1] if pred_dist1 < pred_dist2 else self.dataset_encoding_repos[index2]


        combined_vector = np.array([max_sim], dtype=np.float32)
        decision = selector.predict_method(combined_vector) if not experiment_mode else None
        end_decision_time =  time.time()
        decision_time = (end_decision_time - start_decision_time) * 1000
        
        
        if decision == 1 or experiment_mode:
            print("---- resuing an existing partitioner, EXP MODE: ", experiment_mode)
            matched_encoding_key = self.dataset_encoding_repos[index1].tobytes() if pred_dist1 < pred_dist2 else self.dataset_encoding_repos[index2].tobytes()
          
            matched_partitioner_dataset_name = self.encoding_path_mappings[matched_encoding_key]
            matched_partitioner_dataset_name = matched_partitioner_dataset_name.split("datasets/")[-1]
            matched_partitioner_path = f"stat/partitioner/{matched_partitioner_dataset_name.replace('/', '@')}"
            
            join_para = {"joinDistance" : dist, "partitioner": "UNIVERSAL_QUADTREE_PRECOMPUTED", "matchedPartitioner": matched_partitioner_path}
            if debug:
                print("    ---- matched partitioner path is ", matched_partitioner_path)
            
            our_join_runtime, join_size = run_join_query(dataset1, dataset2, join_para) if pred_dist1 < pred_dist2 else run_join_query(dataset2, dataset1, join_para)
            our_join_runtime += (match_time + decision_time)
            
            if experiment_mode:
                reuse_runtime = our_join_runtime
                max_sim_score = max_sim
                
            print("        ---- our runtime (reuse) ", our_join_runtime, " join size ", join_size, "matching time is", match_time, "decision time is" , decision_time)


        if decision == 0 or experiment_mode:
            print("---- creating partitioner on the fly , EXP MODE: ", experiment_mode)
            join_para = {"joinDistance" : dist, "partitioner": "UNIVERSAL_QUADTREE_ONTHEFLY"}
            our_join_runtime, join_size = run_join_query(dataset1, dataset2, join_para)
            
            if experiment_mode:
                onthefly_runtime = our_join_runtime
                
            if debug:
                print("         ---- our runtime (on-the-fly)",our_join_runtime, " size", join_size)
                

        if baselines is not None:
            for baseline in baselines:
                join_para = {"joinDistance" : dist, "partitioner": baseline}
                baseline_join_runtime, baseline_join_size = run_join_query(dataset1, dataset2, join_para) 
                baseline_join_runtimes.append(baseline_join_runtime)
                if baseline_join_size != join_size:
                    print(" join size diff ", " baseline ", baseline_join_size, " ours ", join_size)
                    #exit(-1)
                if debug:
                    print("        ---- ", baseline, " runtime ", baseline_join_runtime)
        
        if experiment_mode:
            print("experiment mode, reuse onthefly, max_sim", reuse_runtime, onthefly_runtime, max_sim_score)
            return [reuse_runtime, onthefly_runtime, max_sim_score, join_size]
            
        else:
            return [our_join_runtime, baseline_join_runtimes, match_time, decision_time, join_size, decision]
        


    

