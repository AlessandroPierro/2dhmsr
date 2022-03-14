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
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.dyn4j.dynamics.Settings;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class StarterRL {

  public static void main(String[] args) {
    int nExperiments = Integer.parseInt(args[0]);
    ExecutorService executor = Executors.newFixedThreadPool(10);
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
  }


  public static void runExperiment(String[] args, String id) {

    // Configs
    String shape = "biped-4x3";
    String sensorConfig = "uniform-a+vxy-0";
    String rlSensorConfig = "uniform-a-0";
    int nClusters = 4;
    double controllerStep = 0.25;
    double learningRateDecay = 0.995;
    double explorationRateDecay = 0.995;
    double discountFactor = 0.99;
    int seed = 5;

    // Create the body
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig)
        .apply(RobotUtils.buildShape(shape));

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction = new DifferentialRewardFunction();

    // Initialize controller
    ClusteredRLController rlController = new ClusteredRLController(body, rlSensorConfig, null, rewardFunction);

    // Compute sensor readings dimension
    int sensorDimension = rlController.getReadingsDimension();

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(nClusters * sensorDimension);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.45);

    // Create Tabular Q-Learning agent
    ExpectedSARSAAgent rlAgentDiscrete = new ExpectedSARSAAgent(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
        (int) Math.pow(2, sensorDimension * nClusters),
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

    Locomotion locomotion = new Locomotion(15000, terrain, 1000, new Settings());
    locomotion.apply(robot, null);

    locomotion = new Locomotion(1000, terrain, 1000, new Settings());
    GridOnlineViewer.run(
        locomotion,
        Grid.create(1, 1, new NamedValue<>("Clustered RL Controller", robot)),
        Drawers::basicWithMiniWorldAndRL
    );
  }
}
