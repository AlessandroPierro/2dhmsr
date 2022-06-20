package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.behavior.PoseUtils;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.dqn.*;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class StarterRL {

  static final String RESULTS_PATH = "/home/alessandropierro/experiments/";

  public static double[][] createTerrain() {
    return new double[][]{new double[]{0, 10d, 10000d - 10d, 10000d}, new double[]{100d, 5, 5, 100d}};
  }


  static void runBipedDQN() throws IOException {

    // Create a biped
    Grid<Boolean> morphology = RobotUtils.buildShape("biped-4x3");
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-ax+t+r+vxy-0").apply(morphology);

    // Configure observation function
    List<String> sensors = new ArrayList<>();
    sensors.add("y");
    sensors.add("vy");
    sensors.add("a");
    sensors.add("r");
    List<Grid.Key> keys = new ArrayList<>();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        if (j > 0 || (i == 0 || i == 3)) {
          keys.add(new Grid.Key(i, j));
        }
      }
    }
    final int stepsObservation = 4;
    ObservationFunction observationFunction = new ObservationFunction(sensors, keys, stepsObservation);

    System.out.println("Observation function initialized with " + sensors.size() + " sensors, " + keys.size() + " keys and " + stepsObservation + " steps => " + observationFunction.getOutputDimension() + " dimensions");
    System.out.println("Used sensors: " + sensors);

    // Configure reward function
    final int xReward = 1;
    final int yReward = 1;
    RewardFunction rewardFunction = new RewardFunction(xReward, yReward);

    System.out.println("Reward function initialized with x : " + xReward + " and y : " + yReward);

    // Configure DQN agent
    final int inputDimension = observationFunction.getOutputDimension();
    final int outputDimension = 32;
    ContinuousRL agent = new DQNAgent(inputDimension, outputDimension);

    // Configure the control function
    Set<Set<Grid.Key>> clusteredKeys = PoseUtils.computeClusteredByPositionPoses(morphology, 5, 42);
    List<List<Grid.Key>> keysList = new ArrayList<>();
    for (Set<Grid.Key> clusteredKey : clusteredKeys) {
      keysList.add(new ArrayList<>(clusteredKey.stream().toList()));
    }
    ControlFunction controlFunction = new ControlFunction(keysList, 4, 3);

    // Configure robot
    final double controllerFreq = 0.25;
    final RLController controller = new RLController(observationFunction, rewardFunction, agent, controlFunction);
    final StepController stepController = new StepController(controller, controllerFreq);
    final Robot robot = new Robot(stepController, body);

    System.out.println("Robot initialized");

    // Launch learning
    final int episodes = 200;
    final double episodeLength = 100d;

    for (int i = 0; i < episodes; i++) {
      System.out.println("Episode " + i + " of " + episodes + " started");
      // TODO : Introduce early stopping based on rotation
      Locomotion locomotion = new Locomotion(episodeLength, createTerrain(), 5000d, new Settings());
      RLControllerListener listener = new RLControllerListener();
      locomotion.apply(robot, listener);
      File file = new File(RESULTS_PATH + "biped-dqn-episode-" + i + ".csv");
      listener.toFile(file);
    }

    // Test the agent
    System.out.println("Testing the agent");
    final double testEpisodeLength = 60d;
    Locomotion locomotion = new Locomotion(testEpisodeLength, createTerrain(), 5000d, new Settings());
    GridFileWriter.save(
        locomotion,
        robot,
        600,
        400,
        0,
        30,
        VideoUtils.EncoderFacility.JCODEC,
        new File(RESULTS_PATH + "biped-dqn-test.mp4")
    );

    System.out.println("Test finished");
  }

  public static void main(String[] args) throws IOException {
    runBipedDQN();
  }
}
