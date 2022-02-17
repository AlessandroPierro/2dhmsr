package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
import it.units.erallab.hmsrobots.core.controllers.rl.TabularExpectedSARSAAgent;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
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
import java.io.IOException;
import java.util.*;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class ExperimentRL {
  public static void main(String[] args) {
    // Settings
    double learningRate = 0.25;
    double explorationRate = 0.9;
    double learningRateDecay = 1.0;
    double explorationRateDecay = 1.0;
    double discountFactor = 0.7;

    int outputDimension = Integer.parseInt(args[1]);
    int episodes = Integer.parseInt(args[2]);

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+vxy-0.01")
        .apply(RobotUtils.buildShape(args[0]));
    Grid<Boolean> shape = Grid.create(body, Objects::nonNull);

    // Split the robot in 4 cardinal clusters
    Set<Set<Grid.Key>> clusters = computeCardinalPoses(shape);

    ArrayList<ArrayList<Grid.Key>> clustersList = new ArrayList<>();
    int i = 0;
    for (Set<Grid.Key> cluster : clusters) {
      clustersList.add(new ArrayList<>());
      clustersList.get(i).addAll(cluster);
      i++;
    }

    // Create the observation function
    StandardObservationFunction observationFunction = new StandardObservationFunction(body, clustersList);

    // Create input converter
    // TODO Get lower/upper bounds from observation function based on sensors domains
    // TODO How to know the correct number of bins? (touch sensors vs velocity)
    int inputDimension = observationFunction.getOutputDimension();
    double[] binsUpperBound = new double[inputDimension];
    double[] binsLowerBound = new double[inputDimension];
    int[] binsNumber = new int[inputDimension];

    int numberPartitions = 2;

    Arrays.fill(binsUpperBound, 1.0);
    Arrays.fill(binsLowerBound, 0.0);
    Arrays.fill(binsNumber, numberPartitions);

    DiscreteRL.InputConverter standardInputConverter = new EquispacedInputConverter(
        inputDimension,
        binsUpperBound,
        binsLowerBound,
        binsNumber
    );

    // Create output converter
    DiscreteRL.OutputConverter outputConverter;
    outputConverter = new StandardOutputConverter(outputDimension, clustersList, 0.45);

    // Create Random
    Random random = new Random(50);

    // Create QTable initializer
    double averageQ = 0;
    double stdQ = 0;
    Supplier<Double> qtableInitializer = () -> averageQ + stdQ * random.nextGaussian();

    // Instantiate Tabular Q-Learning agent
    TabularExpectedSARSAAgent rlAgentDiscrete = new TabularExpectedSARSAAgent(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor, 50,
        qtableInitializer,
        (int) Math.pow(numberPartitions, inputDimension),
        outputDimension,
        true,
        (int) Math.pow(numberPartitions, inputDimension),
        (int) Math.pow(2, 4)
    );

    // String rlString = SerializationUtils.serialize(rlAgentDiscrete, SerializationUtils.Mode.GZIPPED_JSON);
    // System.out.println(rlString);
    // rlAgentDiscrete = SerializationUtils.deserialize(rlString, TabularExpectedSARSAAgent.class, SerializationUtils.Mode.GZIPPED_JSON);

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(standardInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new AveragedRewardFunction(clustersList, 10);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, clustersList);
    StepController stepController = new StepController(rlController, 0.25);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    Locomotion locomotion;

    // Training episodes
    for (int epochs = 0; epochs < 500; epochs++) {
      for (int j = 0; j < episodes; j++) {
        System.out.println("Training episode " + (j + 1) + "/" + episodes + " on epoch " + (epochs + 1) + "/500");
        locomotion = new Locomotion(200, Locomotion.createTerrain("flat"), new Settings());
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>("phasesRobot", robot)),
            600,
            400,
            0,
            20,
            VideoUtils.EncoderFacility.JCODEC,
            new File(args[3] + "expectedSARSA_" + args[0] + "_"  + epochs + "-" + j + ".mp4"),
            Drawers::basicWithMiniWorld
        );
        System.out.println("Average reward: " + rlController.getAverageReward());
      }

      rlController.stopExploration();

      // Test episodes
      for (int j = 0; j < 2; j++) {
        System.out.println("Testing episode " + (j + 1) + "/2");
        locomotion = new Locomotion(100, Locomotion.createTerrain("flat"), new Settings());
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>("phasesRobot", robot)),
            600,
            400,
            0,
            20,
            VideoUtils.EncoderFacility.JCODEC,
            new File(args[3] + "test_expectedSARSA_" + args[0] + "_" + epochs + "-" + j + ".mp4"),
            Drawers::basicWithMiniWorldAndRL
        );
        System.out.println("Average reward: " + rlController.getAverageReward());
      }

      rlController.setExplorationRate(0.90);
    }
  }

}
