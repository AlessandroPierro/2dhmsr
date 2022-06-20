package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.function.BiFunction;

public class RewardFunction implements Resettable, Serializable, BiFunction<Double, Grid<Voxel>, Double> {

  @JsonProperty
  private final int x;
  @JsonProperty
  private final int y;

  private transient double previousX = Double.MAX_VALUE;

  public RewardFunction(@JsonProperty("x") int x, @JsonProperty("y") int y) {
    this.x = x;
    this.y = y;
  }

  @Override
  public Double apply(Double aDouble, Grid<Voxel> voxels) {
    //final double velocity = voxels.get(x, y).getLinearVelocity().x();
    //final double rotation = voxels.get(x, y).getAngle();
    //return -0.5 * PI < rotation && rotation < 0.5 * PI ? (velocity < 0 ? -50d : velocity * 10d) : -100d;
    final double currentX = voxels.get(x, y).center().x();
    final double reward = previousX == Double.MAX_VALUE ? 0d : Math.exp(currentX - previousX);
    previousX = currentX;
    return reward;
  }

  @Override
  public void reset() {

  }
}
