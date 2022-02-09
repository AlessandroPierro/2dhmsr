package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.function.ToDoubleFunction;

class StandardRewardFunction implements ToDoubleFunction<Grid<Voxel>> {
  /**
   * Reward function based on the x velocity of each voxel in the robot.
   *
   * <p>When applied, the function returns the average x velocity of the voxels, minus the baseline (0 by default).</p>
   */

  private final ArrayList<ArrayList<Grid.Key>> clusters;
  private double baseline;

  private double previousReward = 0;

  StandardRewardFunction(ArrayList<ArrayList<Grid.Key>> clusters, double baseline) {
    this.clusters = clusters;
    this.baseline = baseline;
  }

  StandardRewardFunction(ArrayList<ArrayList<Grid.Key>> clusters) {
    this(clusters, 0);
  }

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {
    previousReward = 0;
    int counter = 0;
    for (ArrayList<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        previousReward += voxels.get(key.x(), key.y()).getSensors().get(1).getReadings()[0];
        ++counter;
      }
    }
    previousReward = previousReward / counter - baseline;
    return previousReward;
  }
}