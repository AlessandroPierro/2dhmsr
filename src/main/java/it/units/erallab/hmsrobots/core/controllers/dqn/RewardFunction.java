package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.function.BiFunction;
import java.util.function.Function;

import static java.lang.Math.PI;

public class RewardFunction implements Resettable, Serializable, BiFunction<Double, Grid<Voxel>, Double> {

  @JsonProperty
  private final int x;
  @JsonProperty
  private final int y;

  public RewardFunction(@JsonProperty("x") int x, @JsonProperty("y") int y) {
    this.x = x;
    this.y = y;
  }

  @Override
  public Double apply(Double aDouble, Grid<Voxel> voxels) {
    final double velocity = voxels.get(x, y).getLinearVelocity().x();
    final double rotation = voxels.get(x, y).getAngle();
    return - 0.75 * PI < rotation && rotation < 0.75 * PI ? 10d * velocity : - 100d;
  }

  @Override
  public void reset() {

  }
}
