package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Objects;
import java.util.function.ToDoubleFunction;

public class DifferentialRewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  private double previousVelocity = 0d;

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {

    double currentVelocity = voxels.stream().filter(Objects::nonNull)
        .map(Grid.Entry::value).filter(Objects::nonNull)
        .mapToDouble(s -> s.getLinearVelocity().x()).sum();
    currentVelocity /= voxels.count(Objects::nonNull);

    double currentReward = currentVelocity - previousVelocity;
    previousVelocity = currentVelocity;

    return currentVelocity;
  }

  @Override
  public void reset() {
    previousVelocity = 0d;
  }
}