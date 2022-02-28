package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.AveragedRewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredObservationFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
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
    int trainingTime = Integer.parseInt(args[1]);
    int testingTime = Integer.parseInt(args[2]);
    String path = args[3];

    // Settings
    String name = "ExpectedSARSA";
    double learningRateDecay = 0.501;
    double explorationRateDecay = 0.501;
    double discountFactor = 0.99;

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+t+vx-0")
        .apply(RobotUtils.buildShape(robotShape));
    Grid<Boolean> shape = Grid.create(body, Objects::nonNull);

    // Compute outputDimension
    int outputDimension = shape.stream().mapToInt(s -> s.value() ? 1 : 0).sum();

    // Create the list of sensors to be used by RL
    Set<Class<? extends Sensor>> usedSensors = new HashSet<>();
    usedSensors.add(AreaRatio.class);
    usedSensors.add(Touch.class);

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

    int inputDimension = observationFunction.getOutputDimension();

    // Create binary input converter
    double[] splitValues = new double[inputDimension];
    Arrays.fill(splitValues, 0.5);
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(inputDimension, splitValues);

    // Create binary output converter
    DiscreteRL.OutputConverter outputConverter;
    outputConverter = new BinaryOutputConverter(outputDimension, clustersList, 0.5);

    // Create Random
    Random random = new Random(42);

    // Create QTable initializer
    double averageQ = 0.0;
    double stdQ = 0.0;
    Supplier<Double> qtableInitializer = () -> averageQ + stdQ * random.nextGaussian();

    // Instantiate Tabular Q-Learning agent
    ExpectedSARSAAgent rlAgentDiscrete = new ExpectedSARSAAgent(
        learningRateDecay,
        explorationRateDecay,
        discountFactor, 50,
        qtableInitializer,
        (int) Math.pow(2, inputDimension),
        (int) Math.pow(2, 4)
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new AveragedRewardFunction(clustersList, 4);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, clustersList);
    StepController stepController = new StepController(rlController, 0.25);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    System.out.println("AverageReward,MinReward,MaxReward");

    // Train period
    Locomotion locomotion = new Locomotion(trainingTime, Locomotion.createTerrain("flat"), new Settings());
    RLListener listener = new RLListener();
    locomotion.apply(robot, listener);

    double averageReward = rlController.getAverageReward();
    double minReward = rlController.getMinReward();
    double maxReward = rlController.getMaxReward();

    System.out.println(averageReward + "," + minReward + "," + maxReward);

    // Test period (exploration rate = 0.05)
    locomotion = new Locomotion(testingTime, Locomotion.createTerrain("flat"), new Settings());
    GridFileWriter.save(
        locomotion,
        Grid.create(1, 1, new NamedValue<>(robotShape + " - " + name + " (test)", robot)),
        640,
        320,
        0,
        20,
        VideoUtils.EncoderFacility.JCODEC,
        new File(path + "test_" + name + "_" + robotShape + ".mp4"),
        Drawers::basicWithMiniWorldAndRL
    );

    averageReward = rlController.getAverageReward();
    minReward = rlController.getMinReward();
    maxReward = rlController.getMaxReward();

    System.out.println(averageReward + "," + minReward + "," + maxReward);

    // Save RL Agent to file
    String rlString = SerializationUtils.serialize(rlAgentDiscrete, SerializationUtils.Mode.JSON);
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(path + name + "_" + robotShape + ".json"));
      writer.write(rlString);
      writer.close();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }
}
