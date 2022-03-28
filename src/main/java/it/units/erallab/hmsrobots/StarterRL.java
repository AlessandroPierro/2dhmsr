package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredRLController;
import it.units.erallab.hmsrobots.core.controllers.rl.RLListener;
import it.units.erallab.hmsrobots.core.controllers.rl.VelocityRewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.QLearningAgent;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.AreaRatio;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.sensors.Touch;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class StarterRL {

  public static void main(String[] args) {
    ExecutorService executor = Executors.newFixedThreadPool(24);
    List<Callable<Integer>> callables = new ArrayList<>();
    double controllerStep = Double.parseDouble(args[0]);

    callables.addAll(IntStream.range(1, 11).mapToObj(seed -> (Callable<Integer>) () -> {
      runExperiment(seed, "worm-5x2", true, false, 0, 2d, controllerStep);
      return seed;
    }).toList());

    callables.addAll(IntStream.range(1, 11).mapToObj(seed -> (Callable<Integer>) () -> {
      runExperiment(seed, "worm-5x2", false, true, 0, 2d, controllerStep);
      return seed;
    }).toList());

    callables.addAll(IntStream.range(1, 11).mapToObj(seed -> (Callable<Integer>) () -> {
      runExperiment(seed, "biped-4x3", true, false, 0, 2d, controllerStep);
      return seed;
    }).toList());

    callables.addAll(IntStream.range(1, 11).mapToObj(seed -> (Callable<Integer>) () -> {
      runExperiment(seed, "biped-4x3", false, true, 0, 2d, controllerStep);
      return seed;
    }).toList());

    callables.add(() -> {
      runExperiment(0, "biped-4x3", true, false, 0, 0d, controllerStep);
      return 0;
    });

    callables.add(() -> {
      runExperiment(0, "biped-4x3", false, true, 0, 0d, controllerStep);
      return 0;
    });

    callables.add(() -> {
      runExperiment(0, "worm-5x2", true, false, 0, 0d, controllerStep);
      return 0;
    });

    callables.add(() -> {
      runExperiment(0, "worm-5x2", false, true, 0, 0d, controllerStep);
      return 0;
    });

    // Execute experiments
    try {
      for (Future<Integer> future : executor.invokeAll(callables)) {
        future.get();
      }
    } catch (ExecutionException | InterruptedException e) {
      e.printStackTrace();
    }

    executor.shutdownNow();
  }


  public static void runExperiment(
      int seed, String shape, boolean touch, boolean area, double mean, double interval, double controllerStep
  ) {

    // Configs
    String sensorConfig = "uniform-a+t+vxy-0";
    int nClusters = 4;
    double learningRateDecay = 0.8794;
    double c = 1.29;
    double discountFactor = 0.32;

    // Create the body and the clusters
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
    Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    // Create the sensor mapping for the observation function
    LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map = new LinkedHashMap<>();
    for (List<Grid.Key> cluster : clusters) {
      LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMapping = new LinkedHashMap<>();
      ToDoubleFunction<double[]> meanOp = value -> value.length == 0 ? 0d : Arrays.stream(value).sum() / value.length;
      ToDoubleFunction<double[]> max = value -> Arrays.stream(value).max().orElse(0d);
      if (area)
        sensorMapping.put(AreaRatio.class, meanOp);
      if (touch)
        sensorMapping.put(Touch.class, max);
      map.put(cluster, sensorMapping);
    }

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction = new VelocityRewardFunction();
    int stateSpaceDimension = (int) Math.pow(2, map.values().stream().mapToInt(LinkedHashMap::size).sum());

    // Initialize controller
    ClusteredRLController rlController = new ClusteredRLController(clusters, map, null, rewardFunction);

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(4);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.5);

    // Create Tabular Q-Learning agent
    QLearningAgent rlAgentDiscrete = new QLearningAgent(learningRateDecay,
        discountFactor,
        c,
        seed,
        mean,
        interval,
        stateSpaceDimension,
        (int) Math.pow(2, nClusters)
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);
    rlController.setRL(rlAgent);

    // Create the RL controller and apply it to the body
    StepController stepController = new StepController(rlController, controllerStep);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    double TERRAIN_BORDER_WIDTH = 10d;
    double TERRAIN_BORDER_HEIGHT = 100d;
    int TERRAIN_LENGTH = 100000;

    double[][] terrain = new double[][]{new double[]{0, TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH - TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH}, new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}};

    Locomotion locomotion = new Locomotion(5000, terrain, 25000, new Settings());
    RLListener listener = new RLListener();
    locomotion.apply(robot, listener);

    File file = new File(shape + "-" + (touch ? "t" : "") + (area ? "a" : "") + "-" + seed + "-meanQ=" + new DecimalFormat(
        "#.0#").format(mean) + "-intervalQ=" + new DecimalFormat("#.0#").format(interval) + "-freq=" + new DecimalFormat(
        "#.0d#").format(controllerStep) + ".csv");
    listener.toFile(file);
  }
}
