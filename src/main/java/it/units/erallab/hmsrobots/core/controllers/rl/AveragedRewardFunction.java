package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.sensors.Velocity;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;

public class AveragedRewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  private final List<List<Grid.Key>> clusters;
  private final double[] rewards;
  private int step = 0;

  public AveragedRewardFunction(List<List<Grid.Key>> clusters, int windowSize) {
    this.clusters = clusters;
    this.rewards = new double[windowSize];
    Arrays.fill(this.rewards, 0);
  }

  public AveragedRewardFunction(List<List<Grid.Key>> clusters) {
    this(clusters, 1);
  }

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {
    int counter = 0;

    double currentReward = 0.0;
    for (List<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        for (Sensor sensor : voxels.get(key.x(), key.y()).getSensors()) {
          while (sensor instanceof CompositeSensor innerSensor) {
            sensor = innerSensor.getSensor();
          }
          if (sensor instanceof Velocity velocitySensor) {
            currentReward += velocitySensor.getReadings()[0];
            ++counter;
          }
        }
      }
    }

    currentReward = currentReward / counter;
    currentReward = currentReward > 0 ? rewards.length * currentReward : currentReward;

    rewards[step] = currentReward / rewards.length;
    step = (step + 1) % rewards.length;

    double totalReward = 0;
    for (double reward : rewards) {
      totalReward += reward;
    }

    return totalReward;
  }

  @Override
  public void reset() {
    Arrays.fill(rewards, 0);
  }
}