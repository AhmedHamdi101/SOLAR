package com.myproject;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sedona.core.spatialRDD.*;
import org.apache.sedona.core.enums.*;
import org.apache.sedona.common.enums.FileDataSplitter;
import org.apache.sedona.core.spatialOperator.*;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.Optional;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.Point;
import org.apache.sedona.common.geometryObjects.Circle;
import org.json.JSONObject;
import org.json.JSONArray;
import scala.Tuple2;
import org.apache.spark.api.java.function.Function2;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileWriter;
import java.io.File;
import java.util.*;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.sedona.core.spatialPartitioning.*;
import org.apache.sedona.core.spatialPartitioning.quadtree.StandardQuadTree;

public class SedonaApp {
    public static void main(String[] args) throws Exception {
        //Logger.getLogger("org.apache.spark").setLevel(Level.DEBUG);
        //Logger.getLogger("com.myproject").setLevel(Level.DEBUG);
		
		long startTime = System.currentTimeMillis();
		long endTime = -1;
		long count = -1;

        SparkSession sparkSession = SparkSession.builder()
                                                .appName("SedonaApp")
                                                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                                                .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
                                                .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(sparkSession.sparkContext());


        String jsonData = new String(Files.readAllBytes(Paths.get("stat/temp/config.json")));
        JSONObject obj = new JSONObject(jsonData);
	String defaultFS = sc.hadoopConfiguration().get("fs.defaultFS");

        String inputLocation1 = defaultFS + obj.getString("inputLocation1");
        String inputLocation2 = defaultFS + obj.getString("inputLocation2");
		String partitionerType = obj.getString("partitioner");
		double joinDistance = obj.getDouble("joinDistance");
		
		
		/*
			we need to consider the following:
		
			"UNIVERSAL_QUADTREE_ONTHEFLY" :       run partitioning and join using universal envelope + store the partitioner
			"UNIVERSAL_QUADTREE_PRECOMPUTED" :    run join using a precomputed partitioner
			
		*/
		
		
		if(partitionerType.equals("UNIVERSAL_QUADTREE_ONTHEFLY")) // run quadtree universal partitioning + store the partitioner
		{
			PointRDD S = new PointRDD(sc, inputLocation1,0, FileDataSplitter.CSV, true, true);
			PointRDD R_pt = new PointRDD(sc, inputLocation2, 0, FileDataSplitter.CSV, true, true);
			CircleRDD R = new CircleRDD(R_pt, joinDistance);
			Object ret = S.userDefinedSpatialPartitioning(GridType.QUADTREE,  null);  // The partitioner here can only be quadtree
			R.spatialPartitioning(S.getPartitioner());
			JavaPairRDD<Geometry, Point> joinedRDD = JoinQuery.DistanceJoinQueryFlat(S, R, true, true);
			
			count = joinedRDD.count();
			endTime = System.currentTimeMillis();
			
			// the following is the serialization of the partitioner to the disk, this does not count as the join execution time because it can be done offline
			String directoryPath = "stat/partitioner/";
			File directory = new File(directoryPath);
			if (!directory.exists()) {
				directory.mkdirs(); // This will create the directory if it doesn't exist
			}

			String filePath = directoryPath + inputLocation1.substring(97).replace("/", "@");
			
			if (ret instanceof StandardQuadTree) {
				StandardQuadTree.serializeQuadTree((StandardQuadTree) ret, filePath);
			} else {
				System.err.println("Expected StandardQuadTree, received " + ret.getClass().getSimpleName());
			}

		}
		
		
		else if(partitionerType.equals("UNIVERSAL_QUADTREE_PRECOMPUTED")) // run join using a precomputed quadtree partitioner
		{
			String matchedPartitioner = obj.getString("matchedPartitioner");

			StandardQuadTree t = StandardQuadTree.deserializeQuadTree(matchedPartitioner);  // No partitioner required here
			PointRDD S = new PointRDD(sc, inputLocation1,0, FileDataSplitter.CSV, true, false);
			PointRDD R_pt = new PointRDD(sc, inputLocation2, 0, FileDataSplitter.CSV, true, false);
			CircleRDD R = new CircleRDD(R_pt, joinDistance);
			S.userDefinedSpatialPartitioning(GridType.QUADTREE,  t);
			R.spatialPartitioning(S.getPartitioner());
			
			//long countS = S.rawSpatialRDD.count();
			//long countR = R.rawSpatialRDD.count(); 
                        JavaPairRDD<Geometry, Point> joinedRDD = JoinQuery.DistanceJoinQueryFlat(S, R, true, true);
			count = joinedRDD.count();
			endTime = System.currentTimeMillis();
		}
		
		
		else  // Run the authentic geospark partitioner
		{	
			PointRDD S = new PointRDD(sc, inputLocation1,0, FileDataSplitter.CSV, true);
			PointRDD R_pt = new PointRDD(sc, inputLocation2, 0, FileDataSplitter.CSV, true);
			CircleRDD R = new CircleRDD(R_pt, joinDistance);
			if(partitionerType.equals("GEOSPARK_QUADTREE"))
			{
				S.spatialPartitioning(GridType.QUADTREE);
			}
			
			
			else if(partitionerType.equals("GEOSPARK_KDBTREE"))
			{
				S.spatialPartitioning(GridType.KDBTREE);
			}		
			
			else if(partitionerType.equals("GEOSPARK_GRID"))
			{
				S.spatialPartitioning(GridType.EQUALGRID);
			}
			
			else
			{
				System.out.println(" partitioner does not exist ");
				System.exit(-1);
			}
			
			
			 
			R.spatialPartitioning(S.getPartitioner());

			//long countS = S.rawSpatialRDD.count();
			//long countR = R.rawSpatialRDD.count();


			JavaPairRDD<Geometry, Point> joinedRDD = JoinQuery.DistanceJoinQueryFlat(S, R, true, true);
			count = joinedRDD.count();
			endTime = System.currentTimeMillis();
			
		}
		

        JSONObject outputJson = new JSONObject();
        outputJson.put("totaltime", endTime - startTime);
	outputJson.put("joinsize", count); //previously count here 
        try (FileWriter file = new FileWriter("stat/temp/output.json")) {
            file.write(outputJson.toString(4));
        }
        


        sc.stop();

    }
}

