package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;

public class ClusteredObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]> {

  private final Set<Class<? extends Sensor>> usedSensors;
  private final List<List<Grid.Key>> clusters;
  private final int numberClusters;
  private final int[] clustersDimensions;
  private final int sensorsDimension;
  private final int outputDimension;

  public ClusteredObservationFunction(
      Grid<Voxel> body,
      Set<Class<? extends Sensor>> usedSensors,
      List<List<Grid.Key>> clusters
  ) {
    this.usedSensors = usedSensors;
    this.clusters = clusters;

    // Computing dimension based on the number of used sensors
    Grid.Key key = clusters.get(0).get(0);
    int size = 0;

    for (Sensor sensor : body.get(key.x(), key.y()).getSensors()) {
      while (sensor instanceof CompositeSensor) {
        sensor = ((CompositeSensor) sensor).getSensor();
      }
      if (usedSensors.contains(sensor.getClass())) {
        size += sensor.getDomains().length;
      }
    }

    this.numberClusters = clusters.size();
    this.sensorsDimension = size;
    this.outputDimension = numberClusters * sensorsDimension;

    // Compute the dimension of each cluster
    this.clustersDimensions = new int[numberClusters];
    for (int i = 0; i < numberClusters; i++) {
      clustersDimensions[i] = clusters.get(i).size();
      if (clustersDimensions[i] == 0) {
        throw new IllegalArgumentException("Voxel clusters with 0 voxels are not permitted!");
      }
    }
  }

  @Override
  public double[] apply(
      Double t, Grid<Voxel> body
  ) {
    double[] output = new double[outputDimension];
    Arrays.fill(output, 0.0);

    // Aggregate the readings for the sensors of each cluster
    for (int i = 0; i < numberClusters; i++) {
      for (Grid.Key key : clusters.get(i)) {
        Voxel voxel = body.get(key.x(), key.y());
        int j = 0;
        for (Sensor sensor : voxel.getSensors()) {
          Sensor sensorTest = sensor;
          while (sensorTest instanceof CompositeSensor) {
            sensorTest = ((CompositeSensor) sensorTest).getSensor();
          }
          if (usedSensors.contains(sensorTest.getClass())) {
            for (double value : sensor.getReadings()) {
              output[i * sensorsDimension + j] += value;
              j += 1;
            }
          }
        }
      }
    }

    // Average readings
    for (int i = 0; i < numberClusters; i++) {
      for (int j = 0; j < sensorsDimension; j++) {
        output[i * sensorsDimension + j] /= clustersDimensions[i];
      }
    }

    return output;
  }

  public int getOutputDimension() {
    return outputDimension;
  }
}