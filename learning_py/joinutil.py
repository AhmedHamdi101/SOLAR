import json
import os
import subprocess
import numpy as np
import pickle
import pandas as pd
import math




def run_join_query(data1_dir, data2_dir, join_para):

    join_para["inputLocation1"], join_para["inputLocation2"] = data1_dir, data2_dir

    # write the join parameter into a json file
    if not os.path.exists('stat/temp/'):
        os.makedirs('stat/temp/', mode = 0o777)

    with open('stat/temp/config.json', 'w') as json_file:
        json.dump(join_para , json_file, indent=4)
        json_file.flush()
        os.fsync(json_file.fileno())


    # Run the spark application to execute the join
    flag = run_spark_java()
    
    if flag == 0:
        # read the post-execution statistic
        with open('stat/temp/output.json', 'r') as join_stat:
            data = json.load(join_stat)
            
        # return the spark join run time and the size of the join
        return (data["totaltime"], data["joinsize"])
        
    else:
        return (math.inf , -1)

    


def run_spark_java():
    try:
        result = subprocess.run(["./yarn_mode.sh"], check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 0
    except subprocess.CalledProcessError as e:
        print("Error during Java program execution:", e.stderr)
        return -1




        





