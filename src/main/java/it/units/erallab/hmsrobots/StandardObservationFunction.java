package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;

class StandardObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]> {

  private final Grid<Voxel> body;
  private final ArrayList<ArrayList<Grid.Key>> clusters;
  private final int numberClusters;
  private final int[] clustersDimensions;
  private final int sensorsDimension;
  private final int outputDimension;

  StandardObservationFunction(
      Grid<Voxel> body,
      ArrayList<ArrayList<Grid.Key>> clusters
  ) {
    this.body = body;
    this.clusters = clusters;

    // Compute the outputDimension
    Grid.Key key = clusters.get(0).get(0);
    AtomicInteger size = new AtomicInteger();
    body.get(key.x(), key.y()).getSensors().forEach(sensor -> size.addAndGet(sensor.getDomains().length));

    this.numberClusters = clusters.size();
    this.sensorsDimension = size.get();
    this.outputDimension = numberClusters * sensorsDimension;

    // Compute the dimension of each cluster
    this.clustersDimensions = new int[numberClusters];
    for (int i = 0; i < numberClusters; i++) {
      clustersDimensions[i] = clusters.get(i).size();
    }
  }

  @Override
  public double[] apply(
      Double t, Grid<Voxel> body
  ) {
    double[] output = new double[16];
    Arrays.fill(output, 0.0);

    // Aggregate the readings for the sensors of each cluster
    for (int i = 0; i < numberClusters; i++) {
      for (Grid.Key key : clusters.get(i)) {
        Voxel voxel = body.get(key.x(), key.y());
        for (Sensor sensor : voxel.getSensors()) {
          int j = 0;
          for (double value : sensor.getReadings()) {
            output[i * sensorsDimension + j] = value;
            j += 1;
          }
        }
      }
    }

    // Average readings
    for (int i = 0; i < numberClusters; i++) {
      for (int j = 0; j < sensorsDimension; j++) {
        output[i*sensorsDimension+j] /= clustersDimensions[i];
      }
    }

    return output;
  }

  int getOutputDimension() {
    return outputDimension;
  }
}