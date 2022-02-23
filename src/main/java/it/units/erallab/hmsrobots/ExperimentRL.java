package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.*;
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
import org.apache.commons.math3.analysis.function.Abs;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.*;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class ExperimentRL {
  public static void main(String[] args) throws IOException {
    // Settings
    double learningRate = 0.1;
    double explorationRate = 0.8;
    double learningRateDecay = 0.99;
    double explorationRateDecay = 0.99;
    double discountFactor = 0.5;

    int outputDimension = Integer.parseInt(args[1]);
    int episodes = Integer.parseInt(args[2]);

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+vxy-0")
        .apply(RobotUtils.buildShape(args[0]));
    Grid<Boolean> shape = Grid.create(body, Objects::nonNull);

    // Create the list of sensors to be used by RL
    // TODO Check for better usage
    Set<Class<? extends Sensor>> usedSensors = new HashSet<>();
    usedSensors.add(AreaRatio.class);

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
    ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(body, usedSensors, clustersList);

    // Create input converter
    // TODO Get lower/upper bounds from observation function based on sensors domains
    // TODO How to know the correct number of bins? (touch sensors vs velocity)
    int inputDimension = observationFunction.getOutputDimension();
    //double[] binsUpperBound = new double[inputDimension];
    //double[] binsLowerBound = new double[inputDimension];
    //int[] binsNumber = new int[inputDimension];

    //int numberPartitions = 2;

    //Arrays.fill(binsUpperBound, 1.0);
    //Arrays.fill(binsLowerBound, 0.0);
    //Arrays.fill(binsNumber, numberPartitions);

    //DiscreteRL.InputConverter standardInputConverter = new EquispacedInputConverter(
    //    inputDimension,
    //    binsUpperBound,
    //    binsLowerBound,
    //    binsNumber
    //);

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(inputDimension, 0.5);

    // Create output converter
    DiscreteRL.OutputConverter outputConverter;
    outputConverter = new BinaryOutputConverter(outputDimension, clustersList, 0.45);

    // Create Random
    Random random = new Random(50);

    // Create QTable initializer
    double averageQ = 0.0;
    double stdQ = 0.0;
    Supplier<Double> qtableInitializer = () -> averageQ + stdQ * random.nextGaussian();

    // Instantiate Tabular Q-Learning agent
    TabularExpectedSARSAAgent rlAgentDiscrete = new TabularExpectedSARSAAgent(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor, 50,
        qtableInitializer,
        true,
        (int) Math.pow(2, inputDimension),
        (int) Math.pow(2, 4)
    );

    // String rlString = SerializationUtils.serialize(rlAgentDiscrete, SerializationUtils.Mode.GZIPPED_JSON);
    // System.out.println(rlString);
    // rlAgentDiscrete = SerializationUtils.deserialize(rlString, TabularExpectedSARSAAgent.class, SerializationUtils.Mode.GZIPPED_JSON);

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new AveragedRewardFunction(clustersList, 10);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, clustersList);
    StepController stepController = new StepController(rlController, 0.5);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    Locomotion locomotion;

    // Training episodes
    for (int epochs = 0; epochs < 50; epochs++) {
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

      double currentExplorationRate = rlAgentDiscrete.getExplorationRate();
      double currentLearningRate = rlAgentDiscrete.getLearningRate();

      rlAgentDiscrete.setExplorationRate(0);
      rlAgentDiscrete.setLearningRate(0);

      // Test episodes
      for (int j = 0; j < 1; j++) {
        System.out.println("Testing episode " + (j + 1) + "/2");
        locomotion = new Locomotion(100, Locomotion.createTerrain("flat"), new Settings());
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>("phasesRobot", robot)),
            660,
            460,
            0,
            20,
            VideoUtils.EncoderFacility.JCODEC,
            new File(args[3] + "test_expectedSARSA_" + args[0] + "_" + epochs + "-" + j + ".mp4"),
            Drawers::basicWithMiniWorldAndRL
        );
        System.out.println("Average reward: " + rlController.getAverageReward());
      }

      rlAgentDiscrete.setExplorationRate(currentExplorationRate);
      rlAgentDiscrete.setLearningRate(currentLearningRate);
    }

    // Serialize agent
    String rlString = SerializationUtils.serialize(rlAgentDiscrete, SerializationUtils.Mode.JSON);
    // save to file
    try {
      FileWriter file = new FileWriter(args[3] + "rlagent.json");
      file.write(rlString);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
