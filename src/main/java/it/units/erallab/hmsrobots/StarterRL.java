package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredRLController;
import it.units.erallab.hmsrobots.core.controllers.rl.DifferentialRewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.QLearningAgent;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
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
    String sensorConfig = "uniform-a+vxy+t-0.01";
    String rlSensorConfig = "uniform-a+t-0.01";
    int nClusters = 4;
    double controllerStep = 0.5;
    double learningRateDecay = 0.99;
    double explorationRateDecay = 0.99;
    double discountFactor = 0.99;
    int seed = 5;

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction = new DifferentialRewardFunction();

    // Initialize controller
    ClusteredRLController rlController = new ClusteredRLController(shape, rlSensorConfig, null, rewardFunction);

    // Compute sensor readings dimension
    int sensorDimension = rlController.getReadingsDimension();

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(nClusters * sensorDimension);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters);

    // Create Tabular Q-Learning agent
    QLearningAgent rlAgentDiscrete = new QLearningAgent(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
        (int) Math.pow(2, sensorDimension),
        (int) Math.pow(2, nClusters)
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);
    rlController.setRL(rlAgent);

    // Create the body
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig)
        .apply(RobotUtils.buildShape(shape));

    // Create the RL controller and apply it to the body
    StepController stepController = new StepController(rlController, controllerStep);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    Locomotion locomotion = new Locomotion(500, Locomotion.createTerrain("flatWithStart-2"), new Settings());
    GridOnlineViewer.run(locomotion, robot);
  }
}
