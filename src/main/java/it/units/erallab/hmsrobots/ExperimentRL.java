package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.SmoothedController;
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
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class ExperimentRL {
  // create main
  public static void main(String[] args) throws IOException {
    // Settings
    // TODO Check right rates
    double learningRate = 0.9;
    double explorationRate = 0.1;
    double learningRateDecay = 0.99;
    double explorationRateDecay = 0.999;
    double discountFactor = 0.99;

    int outputDimension = 1;

    // Create the robot
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-a+vxy+t-0.01")
        .apply(RobotUtils.buildShape("worm-8x3"));

    // Compute shape mask
    Grid<Boolean> shape = Grid.create(body.getW(), body.getH());
    for (Grid.Entry<Voxel> e : body) {
      if (e.value() != null) {
        shape.set(e.key().x(), e.key().y(), true);
      }
    }

    // Split the robot in 4 cardinal clusters
    Set<Set<Grid.Key>> clusters = computeCardinalPoses(shape);
    ArrayList<ArrayList<Grid.Key>> clustersList = new ArrayList<ArrayList<Grid.Key>>();
    int i = 0;
    for (Set<Grid.Key> cluster : clusters) {
      clustersList.add(new ArrayList<Grid.Key>());
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

    Arrays.fill(binsUpperBound, 1.0);
    Arrays.fill(binsLowerBound, -1.0);
    Arrays.fill(binsNumber, 2);

    DiscreteRL.InputConverter standardInputConverter = new EquispacedInputConverter(
        inputDimension,
        binsUpperBound,
        binsLowerBound,
        binsNumber
    );

    // Create output converter
    DiscreteRL.OutputConverter outputConverter;
    outputConverter = new StandardOutputConverter(outputDimension, body, clustersList);

    // Create Random
    Random random = new Random(42);

    // Create QTable initializer
    double averageQ = 1;
    double stdQ = 0.1;
    Supplier<Double> qtableInitializer = () -> {
      return averageQ + stdQ * random.nextGaussian();
    };

    // Instantiate Tabular Q-Learning agent
    TabularExpectedSARSAAgent rlAgentDiscrete = new TabularExpectedSARSAAgent(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor, random,
        qtableInitializer,
        65536,
        24
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(standardInputConverter, outputConverter);

    // Create the reward function
    ToDoubleFunction<Grid<Voxel>> rewardFunction;
    rewardFunction = new StandardRewardFunction(clustersList);

    // Create the RL controller and apply it to the body
    RLController rlController;
    rlController = new RLController(rewardFunction, observationFunction, rlAgent, clustersList);
    SmoothedController smoothedController = new SmoothedController(rlController, 10);
    Robot robot = new Robot(smoothedController, SerializationUtils.clone(body));

    // Launch task
    for (int j = 0; j < 20; j++) {
      System.out.println("Episode " + j);
      Locomotion locomotion = new Locomotion(100, Locomotion.createTerrain("flat"), new Settings());
      GridFileWriter.save(
          locomotion,
          robot,
          600,
          400,
          0,
          20,
          VideoUtils.EncoderFacility.JCODEC,
          new File("/home/eric/experiments/puf-vsr1_smoothed_8x3_" + j + ".mp4")
      );
      rlAgentDiscrete.reset();
    }
  }

}
