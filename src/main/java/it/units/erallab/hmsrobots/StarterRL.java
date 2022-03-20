package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredRLController;
import it.units.erallab.hmsrobots.core.controllers.rl.DifferentialRewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.RLListener;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.ExpectedSARSAAgent;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.AreaRatio;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
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
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class StarterRL {

  public static void main(String[] args) {
    int nExperiments = Integer.parseInt(args[0]);
    int nExecutors = Integer.parseInt(args[1]);
    ExecutorService executor = Executors.newFixedThreadPool(nExecutors);
    List<Callable<Integer>> callables = new ArrayList<>(nExperiments);
    callables.addAll(IntStream.range(1, nExperiments + 1).mapToObj(id -> (Callable<Integer>) () -> {
      runExperiment(args, Integer.toString(id));
      return id;
    }).toList());
    try {
      for (Future<Integer> future : executor.invokeAll(callables)) {
        future.get();
      }
    } catch (ExecutionException | InterruptedException e) {
      e.printStackTrace();
    }

    System.exit(0);
  }


  public static void runExperiment(String[] args, String id) {

    // Configs
    String shape = "biped-4x3";
    String sensorConfig = "uniform-a+vxy-0";
    String rlSensorConfig = "uniform-a-0";
    int nClusters = 4;
    double controllerStep = 0.25;
    double learningRateDecay = 0.5119;
    double explorationRateDecay = 0.8521;
    double discountFactor = 0.7333;
    int seed = Integer.parseInt(id);

    // Create the body and the clusters
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig)
        .apply(RobotUtils.buildShape(shape));
    Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    // Create the sensor mapping for the observation function
    LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map = new LinkedHashMap<>();
    for (List<Grid.Key> cluster : clusters) {
      LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMapping = new LinkedHashMap<>();
      ToDoubleFunction<double[]> mean = value ->
          value.length == 0 ? 0d : Arrays.stream(value).sum() / value.length;
      //ToDoubleFunction<double[]> max = value -> Arrays.stream(value).max().orElse(0d);
      sensorMapping.put(AreaRatio.class, mean);
      //sensorMapping.put(Touch.class, max);
      map.put(cluster, sensorMapping);
    }

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction = new DifferentialRewardFunction();
    int stateSpaceDimension = (int) Math.pow(2, map.values().stream().mapToInt(LinkedHashMap::size).sum());

    // Initialize controller
    ClusteredRLController rlController = new ClusteredRLController(clusters, map, null, rewardFunction);

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(4);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.45);

    // Create Tabular Q-Learning agent
    ExpectedSARSAAgent rlAgentDiscrete = new ExpectedSARSAAgent(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
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

    double[][] terrain = new double[][]{
        new double[]{0, TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH - TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH},
        new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}
    };

    Locomotion locomotion = new Locomotion(10000, terrain, 25000, new Settings());
    RLListener listener = new RLListener(10);
    locomotion.apply(robot, listener);
    File file = new File("log_" + id + ".csv");
    listener.toFile(file);

    locomotion = new Locomotion(60, terrain, 25000, new Settings());

    GridFileWriter.save(
        locomotion,
        Grid.create(1, 1, new NamedValue<>("ExpectedSARSA - seed : " + seed, robot)),
        640,
        320,
        0,
        20,
        VideoUtils.EncoderFacility.JCODEC,
        new File("test_ExpectedSARSA_" + id + ".mp4"),
        Drawers::basicWithMiniWorldAndRL
    );
  }
}
