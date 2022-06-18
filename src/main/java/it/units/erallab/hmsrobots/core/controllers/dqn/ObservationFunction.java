package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

public class ObservationFunction implements BiFunction<Double, Grid<Voxel>, double[]>, Serializable, Resettable {

  @JsonProperty
  protected final List<String> sensors;
  @JsonProperty
  protected final List<Grid.Key> keys;
  @JsonProperty
  protected final int steps;
  @JsonProperty
  protected final int outputDimension;

  protected transient List<Double> observations;

  public ObservationFunction(@JsonProperty("config") List<String> sensors, @JsonProperty("keys") List<Grid.Key> keys, @JsonProperty("steps") int steps) {
    this.sensors = sensors;
    this.keys = keys;
    this.steps = steps;
    this.outputDimension = sensors.size() * keys.size() * steps;
    this.observations = IntStream.range(1, outputDimension + 1).mapToDouble(i -> 0).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
  }

  public ObservationFunction(@JsonProperty("config") List<String> sensors, @JsonProperty("keys") List<Grid.Key> keys) {
    this(sensors, keys, 1);
  }

  @Override
  public double[] apply(Double aDouble, Grid<Voxel> voxels) {
    if (observations == null) {
      observations = IntStream.range(1, outputDimension + 1).mapToDouble(i -> 0).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    for (String sensor : sensors) {
      for (Grid.Key key : keys) {
        observations.add(extractReading(sensor, voxels.get(key.x(), key.y())));
        observations.remove(0);
      }
    }
    return observations.stream().mapToDouble(d -> d).toArray();
  }

  private double extractReading(String sensor, Voxel voxel) {
    switch (sensor) {
      case "x":
        return voxel.center().x();
      case "y":
        return voxel.center().y();
      case "vx":
        return voxel.getLinearVelocity().x();
      case "vy":
        return voxel.getLinearVelocity().y();
      case "a":
        return voxel.getAreaRatio();
      case "r":
        return voxel.getAngle();
    }
    throw new IllegalArgumentException("Observation function - Unknown sensor: " + sensor);
  }

  @Override
  public void reset() {

  }

  public int getOutputDimension() {
    return outputDimension;
  }

}

