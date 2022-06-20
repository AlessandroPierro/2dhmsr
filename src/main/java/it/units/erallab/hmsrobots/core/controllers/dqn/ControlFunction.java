package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;

public class ControlFunction implements BiFunction<Double, double[], Grid<Double>>, Serializable, Resettable {

  @JsonProperty
  protected final List<List<Grid.Key>> keys;
  @JsonProperty
  protected final int w;
  @JsonProperty
  protected final int h;

  public ControlFunction(
      @JsonProperty("keys") List<List<Grid.Key>> keys,
      @JsonProperty("w") int w,
      @JsonProperty("h") int h
  ) {
    this.keys = keys;
    this.w = w;
    this.h = h;
  }

  @Override
  public Grid<Double> apply(Double aDouble, double[] doubles) {
    Grid<Double> grid = Grid.create(w, h);
    int counter = 0;
    for (List<Grid.Key> key : keys) {
      for (Grid.Key k : key) {
        grid.set(k.x(), k.y(), doubles[counter]);
      }
      counter++;
    }
    return grid;
  }

  @Override
  public void reset() {

  }
}
