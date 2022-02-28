package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.sensors.Velocity;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.function.ToDoubleFunction;

public class DifferentialRewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  private final List<List<Grid.Key>> clusters;
  private double previousReward = 0d;

  public DifferentialRewardFunction(List<List<Grid.Key>> clusters) {
    this.clusters = clusters;
  }

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {
    double currentReward = 0d;

    int counter = 0;
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
    double totalReward = currentReward - previousReward;
    previousReward = currentReward;

    return totalReward;
  }

  @Override
  public void reset() {
    previousReward = 0d;
  }
}