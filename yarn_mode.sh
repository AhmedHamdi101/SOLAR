#!/bin/bash

# 1) Get the default filesystem: wasbs://...@....
FS_DEFAULT=$(hdfs getconf -confKey fs.defaultFS)

# 2) The folder in your Blob container where your JAR(s) reside
HDFS_JAR_DIR="/lib2"

# 3) Generate the list of JARs and convert from /lib2/... to 
# wasbs://spatialjoincluster-2025-01-23t08-18-24-613z@spatialjoinclhdistorage.blob.core.windows.net/lib2/...
HDFS_JAR_PATHS=$(
  hdfs dfs -ls "$HDFS_JAR_DIR" \
	    | grep '.jar' \
	      | awk '{print $8}' \
	        | sed "s|^/|${FS_DEFAULT}/|" \
		  | paste -sd "," -
  )

  # Debugging output
  echo "Default FS: $FS_DEFAULT"
  echo "Resolved JARs: $HDFS_JAR_PATHS"

  # Spark-submit with the WASB-based jar paths
  spark-submit \
	    --class com.myproject.SedonaApp \
	      --master yarn \
	        --deploy-mode client \
		  --num-executors 8 \
		    --driver-memory 32g \
		      --executor-memory 12g \
		        --executor-cores 1 \
			  --jars "$HDFS_JAR_PATHS" \
			    SedonaApp.jar

