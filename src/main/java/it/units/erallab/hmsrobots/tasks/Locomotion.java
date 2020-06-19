/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.erallab.hmsrobots.tasks;

import it.units.erallab.hmsrobots.core.objects.*;
import it.units.erallab.hmsrobots.core.objects.immutable.Snapshot;
import it.units.erallab.hmsrobots.util.BoundingBox;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.viewers.SnapshotListener;
import org.apache.commons.lang3.Range;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.Vector2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Locomotion extends AbstractTask<Robot, List<Double>> {

  private final static double INITIAL_PLACEMENT_X_GAP = 3d;
  private final static double INITIAL_PLACEMENT_Y_GAP = 3d;
  private final static double TERRAIN_BORDER_HEIGHT = 100d;
  private final static int TERRAIN_POINTS = 50;

  public enum Metric {
    TRAVEL_X_VELOCITY(false),
    TRAVEL_X_RELATIVE_VELOCITY(false),
    CENTER_AVG_Y(true),
    CONTROL_POWER(true),
    RELATIVE_CONTROL_POWER(true);

    private final boolean toMinimize;

    Metric(boolean toMinimize) {
      this.toMinimize = toMinimize;
    }

    public boolean isToMinimize() {
      return toMinimize;
    }

  }

  private final double finalT;
  private final double[][] groundProfile;
  private final List<Metric> metrics;

  public Locomotion(double finalT, double[][] groundProfile, List<Metric> metrics, Settings settings) {
    super(settings);
    this.finalT = finalT;
    this.groundProfile = groundProfile;
    this.metrics = metrics;
  }

  @Override
  public List<Double> apply(Robot robot, SnapshotListener listener) {
    List<Point2> centerPositions = new ArrayList<>();
    //init world
    World world = new World();
    world.setSettings(settings);
    List<WorldObject> worldObjects = new ArrayList<>();
    Ground ground = new Ground(groundProfile[0], groundProfile[1]);
    ground.addTo(world);
    worldObjects.add(ground);
    //position robot: x of rightmost point is on 2nd point of profile
    BoundingBox boundingBox = robot.boundingBox();
    double xLeft = groundProfile[0][1] + INITIAL_PLACEMENT_X_GAP;
    double yGroundLeft = groundProfile[1][1];
    double xRight = xLeft + boundingBox.max.x - boundingBox.min.x;
    double yGroundRight = yGroundLeft + (groundProfile[1][2] - yGroundLeft) * (xRight - xLeft) / (groundProfile[0][2] - xLeft);
    double topmostGroundY = Math.max(yGroundLeft, yGroundRight);
    Vector2 targetPoint = new Vector2(xLeft, topmostGroundY + INITIAL_PLACEMENT_Y_GAP);
    Vector2 currentPoint = new Vector2(boundingBox.min.x, boundingBox.min.y);
    Vector2 movement = targetPoint.subtract(currentPoint);
    robot.translate(movement);
    //get initial x
    double initCenterX = robot.getCenter().x;
    //add robot to world
    robot.addTo(world);
    worldObjects.add(robot);
    //prepare storage objects
    Grid<Double> lastControlSignals = null;
    Grid<Double> sumOfSquaredControlSignals = Grid.create(robot.getVoxels().getW(), robot.getVoxels().getH(), 0d);
    Grid<Double> sumOfSquaredDeltaControlSignals = Grid.create(robot.getVoxels().getW(), robot.getVoxels().getH(), 0d);
    //run
    double t = 0d;
    while (t < finalT) {
      t = t + settings.getStepFrequency();
      world.step(1);
      robot.act(t);
      //update center position metrics
      centerPositions.add(Point2.build(robot.getCenter()));
      //possibly output snapshot
      if (listener != null) {
        Snapshot snapshot = new Snapshot(t, worldObjects.stream().map(WorldObject::immutable).collect(Collectors.toList()));
        listener.listen(snapshot);
      }
    }
    //compute metrics
    List<Double> results = new ArrayList<>(metrics.size());
    for (Metric metric : metrics) {
      double value = Double.NaN;
      switch (metric) {
        case TRAVEL_X_VELOCITY:
          value = (robot.getCenter().x - initCenterX) / t;
          break;
        case TRAVEL_X_RELATIVE_VELOCITY:
          value = (robot.getCenter().x - initCenterX) / t / Math.max(boundingBox.max.x - boundingBox.min.x, boundingBox.max.y - boundingBox.min.y);
          break;
        case CENTER_AVG_Y:
          value = centerPositions.stream()
              .mapToDouble((p) -> p.y)
              .average()
              .orElse(0);
          break;
        case CONTROL_POWER:
          value = robot.getVoxels().values().stream()
              .filter(v -> (v != null) && (v instanceof ControllableVoxel))
              .mapToDouble(v -> ((ControllableVoxel) v).getControlEnergy())
              .sum() / t;
          break;
        case RELATIVE_CONTROL_POWER:

          value = robot.getVoxels().values().stream()
              .filter(v -> (v != null) && (v instanceof ControllableVoxel))
              .mapToDouble(v -> ((ControllableVoxel) v).getControlEnergy())
              .sum() / t / robot.getVoxels().values().stream().filter(v -> (v != null)).count();
          break;
      }
      results.add(value);
    }
    return results;
  }

  private static double[][] randomTerrain(int n, double length, double peak, double borderHeight, Random random) {
    double[] xs = new double[n + 2];
    double[] ys = new double[n + 2];
    xs[0] = 0d;
    xs[n + 1] = length;
    ys[0] = borderHeight;
    ys[n + 1] = borderHeight;
    for (int i = 1; i < n + 1; i++) {
      xs[i] = 1 + (double) (i - 1) * (length - 2d) / (double) n;
      ys[i] = random.nextDouble() * peak;
    }
    return new double[][]{xs, ys};
  }

  enum TerrainType { FLAT, UNEVEN, STUMP, PIT }

  public static double[][] hardcoreTerrain(double length,
                                           double terrainStartPad,
                                           Range<Double> unevenWidthRange,
                                           double peak,
                                           Range<Double> pitGapRange,
                                           double pitHeight,
                                           Range<Double> stumpWidthRange,
                                           double stumpHeight,
                                           double maxTerrainFlat,
                                           double borderHeight,
                                           Random random) {
    double groundY = 0d;
    boolean newTerrain = true;
    int remaining = (int) Math.ceil(terrainStartPad);
    ArrayList<Double> xs = new ArrayList<>();
    ArrayList<Double> ys = new ArrayList<>();
    TerrainType terrainType = TerrainType.FLAT;

    // add initial border
    xs.add(0d);
    ys.add(borderHeight);
    double prevX = 0d;
    double prevY = borderHeight;

    for (int i = 1; i < Math.round(length); i++) {
      double x = i;
      double y = groundY;

      if (terrainType == TerrainType.UNEVEN) {
        if (newTerrain) {
          // define uneven width
          double min = unevenWidthRange.getMinimum();
          double max = unevenWidthRange.getMaximum();
          double unevenWidth = min + (max - min) * random.nextDouble();
          remaining = (int) Math.ceil(unevenWidth);
        } else {
          // draw peak
          y = random.nextDouble() * peak;
        }
      } else if (terrainType == TerrainType.PIT) {
        if (newTerrain) {
          // define pit gap
          double min = pitGapRange.getMinimum();
          double max = pitGapRange.getMaximum();
          double pitGap = min + (max - min) * random.nextDouble();
          remaining = (int) Math.ceil(pitGap);
        } else {
          // draw pit
          y -= pitHeight;
        }
      } else if (terrainType == TerrainType.STUMP) {
        if (newTerrain) {
          // define stump width
          double min = stumpWidthRange.getMinimum();
          double max = stumpWidthRange.getMaximum();
          double stumpWidth = min + (max - min) * random.nextDouble();
          remaining = (int) Math.ceil(stumpWidth);
        } else {
          // draw stump
          y += stumpHeight;
        }
      }

      if (prevY != y) {
        // ensure to draw initial and final positions of pits/stumps
        if (prevX != xs.get(xs.size() - 1)) {
          xs.add(prevX);
          ys.add(prevY);
        }
        xs.add(x);
        ys.add(y);
      }

      newTerrain = false;
      remaining -= 1;
      if (remaining == 0) {
        double min = maxTerrainFlat / 2;
        double max = maxTerrainFlat;
        remaining = (int) Math.round(min + (max - min) * random.nextDouble());
        if (terrainType == TerrainType.FLAT) {
          int stateIdx = random.nextInt(TerrainType.values().length);
          terrainType = TerrainType.values()[stateIdx];
        } else {
          terrainType = TerrainType.FLAT;
        }
        newTerrain = true;
      }

      prevX = x;
      prevY = y;
    }

    // add final border
    xs.add(length);
    ys.add(borderHeight);

    // convert to double array
    double[] xsArray = new double[xs.size()];
    for (int i = 0; i < xsArray.length; i++) {
      xsArray[i] = xs.get(i);
    }
    // convert to double array
    double[] ysArray = new double[ys.size()];
    for (int i = 0; i < ysArray.length; i++) {
      ysArray[i] = ys.get(i);
    }

    return new double[][]{xsArray, ysArray};
  }

  public static double[][] createTerrain(String name) {
    Random random = new Random(1);
    if (name.equals("flat")) {
      return new double[][]{new double[]{0, 10, 1990, 2000}, new double[]{TERRAIN_BORDER_HEIGHT, 0, 0, TERRAIN_BORDER_HEIGHT}};
    } else if (name.startsWith("uneven")) {
      int h = Integer.parseInt(name.replace("uneven", ""));
      return randomTerrain(TERRAIN_POINTS, 2000, h, TERRAIN_BORDER_HEIGHT, random);
    } else if (name.equals("hardcore")) {
      double terrainStartPad = 44d;
      Range<Double> unevenWidthRange = Range.between(11d, 33d);
      double peak = 1d;
      Range<Double> pitGapRange = Range.between(11d, 33d);
      double pitHeight = 6d;
      Range<Double> stumpWidthRange = Range.between(11d, 33d);
      double stumpHeight = 6d;
      double maxTerrainFlat = 16.5d;

      return hardcoreTerrain(2000,
              terrainStartPad,
              unevenWidthRange,
              peak,
              pitGapRange,
              pitHeight,
              stumpWidthRange,
              stumpHeight,
              maxTerrainFlat,
              TERRAIN_BORDER_HEIGHT,
              random);
    }
    return null;
  }

}
