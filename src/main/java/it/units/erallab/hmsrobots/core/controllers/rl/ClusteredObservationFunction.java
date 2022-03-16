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
import java.util.function.DoubleBinaryOperator;
import java.util.stream.DoubleStream;

public class ClusteredObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]> {

  private final int nClusters;
  private final int nSensorReadings;
  private final LinkedHashMap<List<Grid.Key>, Map<Class<? extends Sensor>, DoubleBinaryOperator>> map;

  public ClusteredObservationFunction(LinkedHashMap<List<Grid.Key>, Map<Class<? extends Sensor>, DoubleBinaryOperator>> map) {
    this.map = map;
    this.nClusters = map.size();
    this.nSensorReadings = map.values().stream().mapToInt(Map::size).sum();
  }


  @Override
  public double[] apply(
      Double t, Grid<Voxel> voxels
  ) {
    double[] observations = new double[nSensorReadings];
    int counter = 0;
    for (Map.Entry<List<Grid.Key>, Map<Class<? extends Sensor>, DoubleBinaryOperator>> entry : map.entrySet()) {
      List<Grid.Key> cluster = entry.getKey();
      Map<Class<? extends Sensor>, DoubleBinaryOperator> sensorMap = entry.getValue();
      for (Map.Entry<Class<? extends Sensor>, DoubleBinaryOperator> sensorEntry : sensorMap.entrySet()) {
        for (Grid.Key key : cluster) {
          List<Sensor> sensors = voxels.get(key.x(), key.y())
              .getSensors()
              .stream()
              .filter(s -> (s instanceof CompositeSensor cs ? cs.getInnermostSensor() : s).getClass()
                  .isAssignableFrom(sensorEntry.getKey())).toList();
          observations[counter] = sensors.stream()
              .flatMapToDouble(s -> DoubleStream.of(Arrays.stream(s.getReadings())
                  .reduce(sensorEntry.getValue())
                  .orElse(0d)))
              .reduce(sensorEntry.getValue())
              .orElse(0.0);
          counter++;
        }
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