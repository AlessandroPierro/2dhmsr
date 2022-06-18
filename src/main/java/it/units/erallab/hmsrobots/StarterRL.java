package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.dqn.*;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.System.exit;

public class StarterRL {

  public static double[][] createTerrain() {
    return new double[][]{new double[]{0, 10d, 10000d - 10d, 10000d}, new double[]{100d, 5, 5, 100d}};
  }


  static void runBipedDQN() throws IOException {

    // Create a biped
    Grid<Boolean> morphology = RobotUtils.buildShape("biped-4x3");
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniformAll-0").apply(morphology);

    // Configure observation function
    List<String> sensors = new ArrayList<>();
    sensors.add("y");
    sensors.add("vx");
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
    final int outputDimension = 10;
    ContinuousRL agent = new DQNAgent(inputDimension, outputDimension);

    // Configure the control function
    ControlFunction controlFunction = new ControlFunction(keys, 4, 3);

    // Configure robot
    final double controllerFreq = 0.2;
    final RLController controller = new RLController(observationFunction, rewardFunction, agent, controlFunction);
    final StepController stepController = new StepController(controller, controllerFreq);
    final Robot robot = new Robot(stepController, body);

    System.out.println("Robot initialized");

    // Run the simulation
    final int episodes = 10;
    final double episodeLength = 100d;

    for (int i = 0; i < episodes; i++) {
      // TODO : implement listener
      System.out.println("Episode " + i + " of " + episodes + " started");
      Locomotion locomotion = new Locomotion(episodeLength, createTerrain(), 5000d, new Settings());
      locomotion.apply(robot, null);
      File file = new File("biped-dqn-episode-" + i + ".csv");

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
        new File("/home/alessandro/experiments/biped-dqn-test.mp4")
    );

    System.out.println("Test finished");
  }

  public static void main(String[] args) throws IOException {
    runBipedDQN();
  }
}
