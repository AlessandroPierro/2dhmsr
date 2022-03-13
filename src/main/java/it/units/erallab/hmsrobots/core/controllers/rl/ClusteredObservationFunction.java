package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ClusteredObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]> {

  private final List<List<Grid.Key>> clusters;
  private final SensorsFilter sensorFilter;

  private final int nClusters;
  private final int nSensorReadings;

  public ClusteredObservationFunction(
      List<List<Grid.Key>> clusters, String sensorConfig
  ) {
    this.clusters = clusters;
    this.sensorFilter = new SensorsFilter(sensorConfig);
    this.nClusters = clusters.size();
    this.nSensorReadings = sensorFilter.getReadingsDimension();
  }

  private static class SensorsFilter implements Function<Sensor, Boolean> {

    private final int readingsDimension;
    Set<String> sensorsType;

    SensorsFilter(String config) {
      Grid<Voxel> testBody = RobotUtils.buildSensorizingFunction(config).apply(RobotUtils.buildShape("box-1x1"));
      this.sensorsType = testBody.get(0, 0)
          .getSensors()
          .stream()
          .filter(Objects::nonNull)
          .map(s -> CompositeSensor.class.isAssignableFrom(s.getClass()) ?
              ((CompositeSensor) s).getInnermostSensor() : s)
          .map(s -> s.getClass().getName())
          .collect(Collectors.toSet());
      this.readingsDimension = testBody.get(0, 0)
          .getSensors()
          .stream()
          .filter(Objects::nonNull)
          .mapToInt(s -> s.getDomains().length)
          .sum();
    }

    @Override
    public Boolean apply(Sensor sensor) {
      sensor = CompositeSensor.class.isAssignableFrom(sensor.getClass()) ?
          ((CompositeSensor) sensor).getInnermostSensor() : sensor;
      return sensorsType.contains(sensor.getClass().getName());
    }

    public int getReadingsDimension() {
      return readingsDimension;
    }
  }

  @Override
  public double[] apply(
      Double t, Grid<Voxel> voxels
  ) {
    double[] observation = new double[nClusters * nSensorReadings];
    Arrays.fill(observation, 0d);

    for (int i = 0; i < nClusters; i++) {
      List<Grid.Key> cluster = clusters.get(i);
      for (Grid.Key key : cluster) {
        Voxel voxel = voxels.get(key.x(), key.y());
        double[] temp = new double[nSensorReadings];
        int k = 0;
        for (Sensor sensor : voxel.getSensors()) {
          if (sensorFilter.apply(sensor)) {
            for (double x : sensor.getReadings()) {
              temp[k] = x / cluster.size();
              k++;
            }
          }
        }
        for (int j = 0; j < nSensorReadings; j++) {
          observation[i * nSensorReadings + j] += temp[j];
        }
      }
    }

    return observation;
  }

  public int getOutputDimension() {
    return nClusters * nSensorReadings;
  }

  public int getnSensorReadings() {
    return nSensorReadings;
  }
}