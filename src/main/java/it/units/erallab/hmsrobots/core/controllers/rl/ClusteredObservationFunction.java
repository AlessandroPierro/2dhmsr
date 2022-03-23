package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.ToDoubleFunction;

public class ClusteredObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]> {

  private final int nClusters;
  private final int nSensorReadings;
  private final LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map;

  public ClusteredObservationFunction(LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map) {
    this.map = map;
    this.nClusters = map.size();
    this.nSensorReadings = map.values().stream().mapToInt(Map::size).sum();
  }


  @Override
  public double[] apply(
      Double t, Grid<Voxel> voxels
  ) {
    // TODO : Clean up
    double[] observations = new double[nSensorReadings];
    Arrays.fill(observations, 0d);
    int counter = 0;

    for (Map.Entry<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> entry : map.entrySet()) {

      List<Grid.Key> cluster = entry.getKey();
      Map<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMap = entry.getValue();

      for (Map.Entry<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorEntry : sensorMap.entrySet()) {

        Class<? extends Sensor> sensorType = sensorEntry.getKey();
        ToDoubleFunction<double[]> aggregationFunction = sensorEntry.getValue();

        double[] temp = new double[cluster.size()];

        for (int i = 0; i < cluster.size(); i++) {
          Grid.Key key = cluster.get(i);
          List<Sensor> sensors = voxels.get(key.x(), key.y())
              .getSensors()
              .stream()
              .filter(s -> (s instanceof CompositeSensor cs ? cs.getInnermostSensor() : s).getClass()
                  .isAssignableFrom(sensorType)).toList();
          temp[i] = aggregationFunction
              .applyAsDouble(sensors.stream().mapToDouble(s -> s.getReadings()[0]).toArray());
        }

        observations[counter] = aggregationFunction.applyAsDouble(temp);
        counter++;

      }

    }
    return observations;
  }

  public int getOutputDimension() {
    return nSensorReadings;
  }

  public int getnClusters() {
    return nClusters;
  }
}