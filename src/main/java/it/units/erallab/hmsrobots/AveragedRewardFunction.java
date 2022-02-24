package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.ToDoubleFunction;

class AveragedRewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  private final ArrayList<ArrayList<Grid.Key>> clusters;
  private final double[] rewards;
  private int step = 0;

  AveragedRewardFunction(ArrayList<ArrayList<Grid.Key>> clusters, int steps) {
    this.clusters = clusters;
    this.rewards = new double[steps];
    Arrays.fill(this.rewards, 0);
  }

  AveragedRewardFunction(ArrayList<ArrayList<Grid.Key>> clusters) {
    this(clusters, 0);
  }

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {
    int counter = 0;

    double currentReward = 0.0;
    for (ArrayList<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        // TODO : Improve this
        if (voxels.get(key.x(), key.y()).getSensors().get(1) instanceof CompositeSensor) {
          currentReward += ((CompositeSensor) voxels.get(key.x(), key.y()).getSensors().get(1)).getSensor()
              .getReadings()[0];
          ++counter;
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