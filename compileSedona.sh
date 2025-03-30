#!/bin/bash

# Step 1: Rebuild Sedona after modifying the source
echo "Rebuilding Sedona..."
cd /home/yliu786/join/sedona/sedona
mvn clean install -DskipTests -Dgeotools -Dspotless.check.skip=true

# Step 2: Update the JAR in HDFS
echo "Updating JAR in HDFS..."
cd /home/yliu786/join/sedona/sedona/spark-shaded/target
hdfs dfs -rm /lib2/sedona-spark-shaded-3.0_2.12-1.6.1.jar
hdfs dfs -put sedona-spark-shaded-3.0_2.12-1.6.1.jar /lib2/

# Step 3: Update the local lib directory
echo "Updating local library..."
cd /home/yliu786/join/SedonaProject/lib
rm sedona-spark-shaded-3.0_2.12-1.6.1.jar
hdfs dfs -get /lib2/sedona-spark-shaded-3.0_2.12-1.6.1.jar

# Step 4: Recompile Java code after modifications
echo "Recompiling Java code..."
cd /home/yliu786/join/
javac -cp "SedonaProject/lib/*" SedonaProject/src/com/myproject/SedonaApp.java -d SedonaProject/bin

# Step 5: Package the Java application into a JAR
echo "Packaging Java application..."
cd SedonaProject
jar cvfe SedonaApp.jar com.myproject.SedonaApp -C bin/ .

# Step 6: Move the JAR to the root directory
echo "Moving JAR to root directory..."
mv SedonaApp.jar ../

echo "All steps completed successfully!"

