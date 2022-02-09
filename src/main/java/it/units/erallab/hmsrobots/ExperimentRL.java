package it.units.erallab.hmsrobots;

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
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
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
  public static void main(String[] args) throws IOException {
    // Settings
    double learningRate = 0.01;
    double explorationRate = 0.9;
    double learningRateDecay = 1.0;
    double explorationRateDecay = 0.95;
    double discountFactor = 0.99;

    int outputDimension = Integer.parseInt(args[1]);
    int episodes = Integer.parseInt(args[2]);

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+vxy-0.01")
        .apply(RobotUtils.buildShape(args[0]));
    Grid<Boolean> shape = Grid.create(body, Objects::nonNull);

    // Split the robot in 4 cardinal clusters
    Set<Set<Grid.Key>> clusters = computeCardinalPoses(shape);
    //clusters.forEach(System.out::println);
    //System.exit(0);

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

    int numberPartitions = 4;

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
    outputConverter = new StandardOutputConverter(outputDimension, clustersList, 0.4);

    // Create Random
    Random random = new Random(42);

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
        discountFactor, random,
        qtableInitializer,
        (int) Math.pow(numberPartitions, inputDimension),
        outputDimension,
        true
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(standardInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new StandardRewardFunction(clustersList);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, 15, clustersList);
    Robot robot = new Robot(rlController, SerializationUtils.clone(body));

    Locomotion locomotion;

    // Launch task
    for (int j = 0; j < episodes; j++) {

      locomotion = new Locomotion(200, Locomotion.createTerrain("flat"), new Settings());
      GridFileWriter.save(
          locomotion,
          Grid.create(1, 1, new NamedValue<>("phasesRobot", robot)),
          600,
          400,
          0,
          20,
          VideoUtils.EncoderFacility.JCODEC,
          new File(args[3] + "expectedSARSA_" + args[0] + "_" + j + ".mp4"),
          Drawers::basicWithMiniWorldAndSpectra
      );
    }
  }

}
