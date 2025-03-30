#!/bin/bash


HDFS_DIR="hdfs:///lib2/"
# List of JAR paths on HDFS (you need to list your JARs manually or use an HDFS command to get them)
# Example: HDFS_JAR_PATHS="hdfs:///user/yliu786/lib2/jar1.jar,hdfs:///user/yliu786/lib2/jar2.jar"
HDFS_PATHS=$(hadoop fs -ls $HDFS_DIR | grep '.jar' | awk '{print $8}' | paste -sd "," -)

# Spark-submit command
spark-submit --class com.myproject.SedonaApp --master local[*] --driver-memory 30g --jars "$HDFS_PATHS" SedonaApp.jar
