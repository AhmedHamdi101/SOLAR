#!/bin/bash

# Step 1: Recompile Java code after modifications
echo "Recompiling Java code..."
cd /home/yliu786/join/
javac -cp "SedonaProject/lib/*" SedonaProject/src/com/myproject/SedonaApp.java -d SedonaProject/bin

# Step 2: Package the Java application into a JAR
echo "Packaging Java application..."
cd SedonaProject
jar cvfe SedonaApp.jar com.myproject.SedonaApp -C bin/ .

# Step 3: Move the JAR to the root directory
echo "Moving JAR to root directory..."
mv SedonaApp.jar ../

echo "Java code recompiled and packaged successfully!"

