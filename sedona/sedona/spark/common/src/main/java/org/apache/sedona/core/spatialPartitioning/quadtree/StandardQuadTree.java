package org.apache.sedona.core.spatialPartitioning.quadtree;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sedona.core.spatialPartitioning.PartitioningUtils;
import org.apache.sedona.common.utils.HalfOpenRectangle;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.Point;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.HashMap;

import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class StandardQuadTree<T> extends PartitioningUtils
        implements Serializable
{
    public static final int REGION_SELF = -1;
    public static final int REGION_NW = 0;
    public static final int REGION_NE = 1;
    public static final int REGION_SW = 2;
    public static final int REGION_SE = 3;

    private final int maxItemsPerZone;
    private final int maxLevel;
    private final int level;

    private final List<QuadNode<T>> nodes = new ArrayList<>();

    public QuadRectangle zone;
    private int nodeNum = 0;

    private StandardQuadTree<T>[] regions;

    public StandardQuadTree(QuadRectangle definition, int level)
    {
        this(definition, level, 5, 10);
    }

    public StandardQuadTree(QuadRectangle definition, int level, int maxItemsPerZone, int maxLevel)
    {
        this.maxItemsPerZone = maxItemsPerZone;
        this.maxLevel = maxLevel;
        this.zone = definition;
        this.level = level;
    }

    public QuadRectangle getZone()
    {
        return this.zone;
    }

    private int findRegion(QuadRectangle r, boolean split)
    {
        int region = REGION_SELF;
        if (nodeNum >= maxItemsPerZone && this.level < maxLevel) {

            if (regions == null && split) {

                this.split();
            }


            if (regions != null) {
                for (int i = 0; i < regions.length; i++) {
                    if (regions[i].getZone().contains(r)) {
                        region = i;
                        break;
                    }
                }
            }
        }

        return region;
    }

    private int findRegion(int x, int y)
    {
        int region = REGION_SELF;

        if (regions != null) {
            for (int i = 0; i < regions.length; i++) {
                if (regions[i].getZone().contains(x, y)) {
                    region = i;
                    break;
                }
            }
        }
        return region;
    }

    private StandardQuadTree<T> newQuadTree(QuadRectangle zone, int level)
    {
        return new StandardQuadTree<T>(zone, level, this.maxItemsPerZone, this.maxLevel);
    }

    private void split()
    {

        regions = new StandardQuadTree[4];

        double newWidth = zone.width / 2;
        double newHeight = zone.height / 2;
        int newLevel = level + 1;

        regions[REGION_NW] = newQuadTree(new QuadRectangle(
                zone.x,
                zone.y + zone.height / 2,
                newWidth,
                newHeight
        ), newLevel);

        regions[REGION_NE] = newQuadTree(new QuadRectangle(
                zone.x + zone.width / 2,
                zone.y + zone.height / 2,
                newWidth,
                newHeight
        ), newLevel);

        regions[REGION_SW] = newQuadTree(new QuadRectangle(
                zone.x,
                zone.y,
                newWidth,
                newHeight
        ), newLevel);

        regions[REGION_SE] = newQuadTree(new QuadRectangle(
                zone.x + zone.width / 2,
                zone.y,
                newWidth,
                newHeight
        ), newLevel);
    }


    public void forceGrowUp(int minLevel)
    {
        if (minLevel < 1) {
            throw new IllegalArgumentException("minLevel must be >= 1. Received " + minLevel);
        }

        split();
        nodeNum = maxItemsPerZone;
        if (level + 1 >= minLevel) {

            return;
        }

        for (StandardQuadTree<T> region : regions) {
            region.forceGrowUp(minLevel);
        }
    }

    public void insert(QuadRectangle r, T element)
    {
        int region = this.findRegion(r, true);
        if (region == REGION_SELF || this.level == maxLevel) {
            nodes.add(new QuadNode<T>(r, element));
            nodeNum++;
            return;
        }
        else {
            regions[region].insert(r, element);
        }

        if (nodeNum >= maxItemsPerZone && this.level < maxLevel) {

            List<QuadNode<T>> tempNodes = new ArrayList<>();
            tempNodes.addAll(nodes);

            nodes.clear();
            for (QuadNode<T> node : tempNodes) {
                this.insert(node.r, node.element);
            }
        }
    }

    public void dropElements()
    {
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                tree.nodes.clear();
                return true;
            }
        });
    }

    public List<T> getElements(QuadRectangle r)
    {
        int region = this.findRegion(r, false);

        final List<T> list = new ArrayList<>();

        if (region != REGION_SELF) {
            for (QuadNode<T> node : nodes) {
                list.add(node.element);
            }

            list.addAll(regions[region].getElements(r));
        }
        else {
            addAllElements(list);
        }

        return list;
    }


    private void traverse(Visitor<T> visitor)
    {
        if (!visitor.visit(this)) {
            return;
        }

        if (regions != null) {
            regions[REGION_NW].traverse(visitor);
            regions[REGION_NE].traverse(visitor);
            regions[REGION_SW].traverse(visitor);
            regions[REGION_SE].traverse(visitor);
        }
    }


    private void traverseWithTrace(VisitorWithLineage<T> visitor, String lineage)
    {
        if (!visitor.visit(this, lineage)) {
            return;
        }

        if (regions != null) {
            regions[REGION_NW].traverseWithTrace(visitor, lineage + REGION_NW);
            regions[REGION_NE].traverseWithTrace(visitor, lineage + REGION_NE);
            regions[REGION_SW].traverseWithTrace(visitor, lineage + REGION_SW);
            regions[REGION_SE].traverseWithTrace(visitor, lineage + REGION_SE);
        }
    }

    private void addAllElements(final List<T> list)
    {
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                for (QuadNode<T> node : tree.nodes) {
                    list.add(node.element);
                }
                return true;
            }
        });
    }

    public boolean isLeaf()
    {
        return regions == null;
    }

    public List<QuadRectangle> getAllZones()
    {
        final List<QuadRectangle> zones = new ArrayList<>();
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                zones.add(tree.zone);
                return true;
            }
        });

        return zones;
    }

    public HashMap<Integer, Envelope> getPartitionEnvelopeMap()
    {
                assignPartitionIds();
        HashMap<Integer, Envelope> partitionEnvelopeMap = new HashMap<>();
                traverse(new Visitor<T>() {
                        @Override
                        public boolean visit(StandardQuadTree<T> tree) {
                                if (tree.isLeaf() && tree.zone.partitionId != null) {
                                        partitionEnvelopeMap.put(tree.zone.partitionId, tree.zone.getEnvelope());
                                }
                                if(tree.isLeaf() && tree.zone.partitionId == null)
                                {
                                     System.out.println("leaf node with null partitionID encountered");
                                     System.exit(0);
                                }



                                return true;
                        }
                });

        return partitionEnvelopeMap;
    }






    public int getTotalNumLeafNode()
    {
        final MutableInt leafCount = new MutableInt(0);
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                if (tree.isLeaf()) {
                    leafCount.increment();
                }
                return true;
            }
        });

        return leafCount.getValue();
    }


    public QuadRectangle getZone(int x, int y)
            throws ArrayIndexOutOfBoundsException
    {
        int region = this.findRegion(x, y);
        if (region != REGION_SELF) {
            return regions[region].getZone(x, y);
        }
        else {
            if (this.zone.contains(x, y)) {
                return this.zone;
            }

            throw new ArrayIndexOutOfBoundsException("[Sedona][StandardQuadTree] this pixel is out of the quad tree boundary.");
        }
    }

    public QuadRectangle getParentZone(int x, int y, int minLevel)
            throws Exception
    {
        int region = this.findRegion(x, y);

        if (level < minLevel) {

            if (region == REGION_SELF) {
                assert regions == null;
                if (zone.contains(x, y)) {

                    throw new Exception("[Sedona][StandardQuadTree][getParentZone] this leaf node doesn't have enough depth. " +
                            "Please check ForceGrowUp. Expected: " + minLevel + " Actual: " + level + ". Query point: " + x + " " + y +
                            ". Tree statistics, total leaf nodes: " + getTotalNumLeafNode());
                }
                else {
                    throw new Exception("[Sedona][StandardQuadTree][getParentZone] this pixel is out of the quad tree boundary.");
                }
            }
            else {
                return regions[region].getParentZone(x, y, minLevel);
            }
        }
        if (zone.contains(x, y)) {
            return zone;
        }

        throw new Exception("[Sedona][StandardQuadTree][getParentZone] this pixel is out of the quad tree boundary.");
    }

    public List<QuadRectangle> findZones(QuadRectangle r)
    {
        final Envelope envelope = r.getEnvelope();

        final List<QuadRectangle> matches = new ArrayList<>();
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                if (!disjoint(tree.zone.getEnvelope(), envelope)) {
                    if (tree.isLeaf()) {
                        matches.add(tree.zone);
                    }
                    return true;
                }
                else {
                    return false;
                }
            }
        });

        return matches;
    }

    private boolean disjoint(Envelope r1, Envelope r2)
    {
        return !r1.intersects(r2) && !r1.covers(r2) && !r2.covers(r1);
    }

    public void assignPartitionIds()
    {
        traverse(new Visitor<T>()
        {
            private int partitionId = 0;

            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                if (tree.isLeaf()) {
                    tree.getZone().partitionId = partitionId;
                    //System.out.println("the id assigned just now is" + tree.zone.partitionId);
                    partitionId++;
                }
                return true;
            }
        });
    }

    public void assignPartitionLineage()
    {
        traverseWithTrace(new VisitorWithLineage<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree, String lineage)
            {
                if (tree.isLeaf()) {
                    tree.getZone().lineage = lineage;
                }
                return true;
            }
        }, "");
    }

    @Override
    public Iterator<Tuple2<Integer, Geometry>> placeObject(Geometry geometry) {
        Objects.requireNonNull(geometry, "spatialObject");

        final Envelope envelope = geometry.getEnvelopeInternal();

        final List<QuadRectangle> matchedPartitions = findZones(new QuadRectangle(envelope));

        final Point point = geometry instanceof Point ? (Point) geometry : null;

        final Set<Tuple2<Integer, Geometry>> result = new HashSet<>();
        for (QuadRectangle rectangle : matchedPartitions) {

            if (point != null && !(new HalfOpenRectangle(rectangle.getEnvelope())).contains(point)) {
                continue;
            }

            result.add(new Tuple2(rectangle.partitionId, geometry));
        }

        return result.iterator();
    }

    @Override
    public Set<Integer> getKeys(Geometry geometry) {
        Objects.requireNonNull(geometry, "spatialObject");

        final Envelope envelope = geometry.getEnvelopeInternal();

        final List<QuadRectangle> matchedPartitions = findZones(new QuadRectangle(envelope));

        final Point point = geometry instanceof Point ? (Point) geometry : null;

        final Set<Integer> result = new HashSet<>();
        for (QuadRectangle rectangle : matchedPartitions) {

            if (point != null && !(new HalfOpenRectangle(rectangle.getEnvelope())).contains(point)) {
                continue;
            }

            result.add(rectangle.partitionId);
        }

        return result;
    }

    @Override
    public List<Envelope> fetchLeafZones() {
        final List<Envelope> leafZones = new ArrayList<>();
        traverse(new Visitor<T>()
        {
            @Override
            public boolean visit(StandardQuadTree<T> tree)
            {
                if (tree.isLeaf()) {
                    leafZones.add(tree.zone.getEnvelope());
                }
                return true;
            }
        });
        return leafZones;
    }
	
	

    private interface Visitor<T>
    {

        boolean visit(StandardQuadTree<T> tree);
    }

    private interface VisitorWithLineage<T>
    {

        boolean visit(StandardQuadTree<T> tree, String lineage);
    }
	
	public static <T> void serializeQuadTree(StandardQuadTree<T> tree, String filePath) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(tree);
        } catch (IOException e) {
            System.err.println("Failed to serialize the QuadTree: " + e.getMessage());
            throw new RuntimeException("Serialization failed", e);
        }
    }

    public static <T> StandardQuadTree<T> deserializeQuadTree(String filePath) {
    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
        Object object = in.readObject();
        if (object instanceof StandardQuadTree<?>) {
            return (StandardQuadTree<T>) object;
        } else {
            throw new IllegalArgumentException("Invalid object type");
        }
    } catch (IOException | ClassNotFoundException e) {
        System.err.println("Failed to deserialize the QuadTree: " + e.getMessage());
        throw new RuntimeException("Deserialization failed", e);
    }
}

}

