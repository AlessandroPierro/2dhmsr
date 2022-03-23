package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Objects;
import java.util.function.ToDoubleFunction;

public class VelocityRewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {

    double currentVelocity = voxels.stream().filter(Objects::nonNull)
        .map(Grid.Entry::value).filter(Objects::nonNull)
        .mapToDouble(s -> s.getLinearVelocity().x()).sum();
    currentVelocity /= voxels.count(Objects::nonNull);

    return currentVelocity;
  }

  @Override
  public void reset() {
    // nothing to do
  }
}