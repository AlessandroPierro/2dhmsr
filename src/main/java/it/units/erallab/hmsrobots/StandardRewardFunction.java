package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.function.ToDoubleFunction;

class StandardRewardFunction implements ToDoubleFunction<Grid<Voxel>> {

  private final ArrayList<ArrayList<Grid.Key>> clusters;

  StandardRewardFunction(ArrayList<ArrayList<Grid.Key>> clusters) {
    this.clusters = clusters;
  }

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {
    double reward = 0;
    for (ArrayList<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        reward += voxels.get(key.x(), key.y()).getSensors().get(1).getReadings()[1] - 1;
      }
    }
    return reward;
  }
}