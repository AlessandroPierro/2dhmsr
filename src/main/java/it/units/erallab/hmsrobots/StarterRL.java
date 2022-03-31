package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredRLController;
import it.units.erallab.hmsrobots.core.controllers.rl.RLListener;
import it.units.erallab.hmsrobots.core.controllers.rl.RewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularSARSALambda;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.Angle;
import it.units.erallab.hmsrobots.core.sensors.AreaRatio;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.sensors.Touch;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class StarterRL {

  private static double[][] getTerrain() {
    double TERRAIN_BORDER_WIDTH = 10d;
    double TERRAIN_BORDER_HEIGHT = 100d;
    int TERRAIN_LENGTH = 1000000;
    return new double[][]{new double[]{0, TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH - TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH}, new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}};
  }

  public static void main(String[] args) {

    int nThreads = Integer.parseInt(args[0]);

    ExecutorService executor = Executors.newFixedThreadPool(nThreads);
    List<Callable<Integer>> callables = new ArrayList<>();

    callables.add(() -> {
      runExperiment(0, "biped-4x3", false, true, 0, 0d, 0.5);
      return 0;
    });

    try {
      for (Future<Integer> future : executor.invokeAll(callables)) {
        future.get();
      }
    } catch (ExecutionException | InterruptedException e) {
      e.printStackTrace();
    }

    executor.shutdown();
  }

  private static void runExperiment(
      int seed, String shape, boolean touch, boolean area, double mean, double interval, double controllerStep
  ) {

    // Configs
    String sensorConfig = "uniform-a+t+r+vxy-0";
    int nClusters = 4;

    // Create the body and the clusters
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
    Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    // Create the sensor mapping for the observation function
    int counter = 0;
    LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> clusterMap = new LinkedHashMap<>();
    for (List<Grid.Key> cluster : clusters) {
      LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMap = new LinkedHashMap<>();
      ToDoubleFunction<double[]> meanOp = x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length;
      ToDoubleFunction<double[]> max = x -> Arrays.stream(x).max().orElse(0d);
      ToDoubleFunction<double[]> min = x -> Arrays.stream(x).min().orElse(0d);
      ToDoubleFunction<double[]> angleMap = x -> 0.25 < min.applyAsDouble(x) && max.applyAsDouble(x) < 0.75 ? 0.45 : 0.85;
      if (area)
        sensorMap.put(AreaRatio.class, meanOp);
      if (touch)
        sensorMap.put(Touch.class, max);
      if (counter == 0) {
        sensorMap.put(Angle.class, angleMap);
        counter++;
      }
      clusterMap.put(cluster, sensorMap);
    }

    // Compute dimensions
    int sensorReadingsDimension = clusterMap.values().stream().mapToInt(LinkedHashMap::size).sum();
    int actionSpaceDimension = (int) Math.pow(2, nClusters);
    int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction = new RewardFunction();

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.5);

    // Create Tabular Q-Learning agent
    TabularSARSALambda rlAgentDiscrete = new TabularSARSALambda(
        0.99,
        0.75,
        seed,
        stateSpaceDimension,
        actionSpaceDimension
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);

    // Create the RL controller and apply it to the body
    ClusteredRLController rlController = new ClusteredRLController(clusters, clusterMap, rlAgent, rewardFunction);
    StepController stepController = new StepController(rlController, 0.5);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    // Run episodes
    Locomotion locomotion;
    RLListener listener;

    for (int i = 0; i < 50; i++) {
      locomotion = new Locomotion(500, getTerrain(), 50000, new Settings());
      listener = new RLListener();
      locomotion.apply(robot, listener);
      File file = new File("episode-" + i + ".csv");
      listener.toFile(file);
    }

    rlAgentDiscrete.stopLearning();

    locomotion = new Locomotion(30, getTerrain(), 50000, new Settings());
    GridFileWriter.save(
        locomotion,
        Grid.create(1, 1, new NamedValue<>("phasesRobot", robot)),
        600, 600,
        0, 20,
        VideoUtils.EncoderFacility.JCODEC,
        new File("video-test.mp4"),
        Drawers::basicWithMiniWorldAndRL
    );

  }
}
