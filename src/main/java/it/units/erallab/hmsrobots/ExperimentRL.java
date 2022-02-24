package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
import it.units.erallab.hmsrobots.core.controllers.rl.TabularQLearningAgent;
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class ExperimentRL {
  public static void main(String[] args) {
    // Command line arguments
    String robotShape = args[0];
    // TODO : Automate outputDimension counting
    int outputDimension = Integer.parseInt(args[1]);
    int epochs = Integer.parseInt(args[2]);
    int trainEpisodes = Integer.parseInt(args[3]);
    int testEpisodes = Integer.parseInt(args[4]);
    String path = args[5];

    // Settings
    double learningRate = 0.1;
    double explorationRate = 0.8;
    double learningRateDecay = 0.995;
    double explorationRateDecay = 0.995;
    double discountFactor = 0.85;

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+vxy-0")
        .apply(RobotUtils.buildShape(robotShape));
    Grid<Boolean> shape = Grid.create(body, Objects::nonNull);

    // Create the list of sensors to be used by RL
    // TODO Check for better usage
    Set<Class<? extends Sensor>> usedSensors = new HashSet<>();
    usedSensors.add(AreaRatio.class);

    // Split the robot in 4 cardinal clusters
    Set<Set<Grid.Key>> clusters = computeCardinalPoses(shape);

    List<List<Grid.Key>> clustersList = new ArrayList<>();
    for (Set<Grid.Key> cluster : clusters) {
      clustersList.add(new ArrayList<>(cluster));
    }

    // Create the observation function
    ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(
        body,
        usedSensors,
        clustersList
    );

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
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(inputDimension, 0.45);

    // Create output converter
    DiscreteRL.OutputConverter outputConverter;
    outputConverter = new BinaryOutputConverter(outputDimension, clustersList, 0.5);

    // Create Random
    Random random = new Random(50);

    // Create QTable initializer
    double averageQ = 0.0;
    double stdQ = 0.0;
    Supplier<Double> qtableInitializer = () -> averageQ + stdQ * random.nextGaussian();

    // Instantiate Tabular Q-Learning agent
    TabularQLearningAgent rlAgentDiscrete = new TabularQLearningAgent(
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

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new AveragedRewardFunction(clustersList, 5);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, clustersList);
    StepController stepController = new StepController(rlController, 0.45);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    Locomotion locomotion;

    double currentExplorationRate = explorationRate;
    double currentLearningRate = learningRate;

    System.out.println("Epoch,Episode,Kind,AverageReward,MinReward,MaxReward,ExplorationRate,LearningRate");

    for (int i = 1; i <= epochs; ++i) {
      // Training episodes
      for (int j = 1; j <= trainEpisodes; j++) {
        locomotion = new Locomotion(200, Locomotion.createTerrain("flat"), new Settings());
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>(robotShape + " - QLearning (train)", robot)),
            512,
            256,
            0,
            15,
            VideoUtils.EncoderFacility.JCODEC,
            new File(path + "QLearning_" + robotShape + "_" + i + "-" + j + ".mp4"),
            Drawers::basicWithMiniWorld
        );

        double averageReward = rlController.getAverageReward();
        double minReward = rlController.getMinReward();
        double maxReward = rlController.getMaxReward();

        currentExplorationRate = rlAgentDiscrete.getExplorationRate();
        currentLearningRate = rlAgentDiscrete.getLearningRate();

        System.out.println(i + "," + j + "," + "train" + "," + averageReward + "," + minReward + "," + maxReward + "," + currentExplorationRate + "," + currentLearningRate);
      }

      rlAgentDiscrete.setExplorationRate(0);
      rlAgentDiscrete.setLearningRate(0);

      // Test episodes (no exploration)
      for (int j = 1; j <= testEpisodes; j++) {
        locomotion = new Locomotion(100, Locomotion.createTerrain("flat"), new Settings());
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>(robotShape + " - QLearning (test)", robot)),
            640,
            320,
            0,
            20,
            VideoUtils.EncoderFacility.JCODEC,
            new File(path + "test_QLearning_" + robotShape + "_" + i + "-" + j + ".mp4"),
            Drawers::basicWithMiniWorldAndRL
        );

        double averageReward = rlController.getAverageReward();
        double minReward = rlController.getMinReward();
        double maxReward = rlController.getMaxReward();

        System.out.println(i + "," + j + "," + "test" + "," + averageReward + "," + minReward + "," + maxReward + ",0,0");
      }

      rlAgentDiscrete.setExplorationRate(currentExplorationRate);
      rlAgentDiscrete.setLearningRate(currentLearningRate);

      String rlString = SerializationUtils.serialize(rlAgentDiscrete, SerializationUtils.Mode.JSON);
      try {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path + "QLearning_" + robotShape + "_" + i + ".json"));
        writer.write(rlString);
        writer.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }
}
