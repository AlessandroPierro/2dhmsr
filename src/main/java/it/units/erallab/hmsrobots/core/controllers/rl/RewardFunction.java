package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.function.ToDoubleFunction;

public class RewardFunction implements ToDoubleFunction<Grid<Voxel>>, Resettable {

  private double previousPosition = Double.NEGATIVE_INFINITY;

  @Override
  public double applyAsDouble(Grid<Voxel> voxels) {

    if (previousPosition == Double.NEGATIVE_INFINITY) {
      previousPosition = voxels.get(0, 0).center().x();
    }

    double rotation = voxels.get(0, 0).getAngle();
    double currentPosition = voxels.get(0, 0).center().x();
    double deltaPosition = currentPosition - previousPosition;

    double reward = rotation < -Math.PI / 2 || rotation > Math.PI / 2 ?
        -100d : (deltaPosition <= 0d ? -50d : 10 * deltaPosition);

    previousPosition = currentPosition;
    return reward;
  }

  @Override
  public void reset() {
    previousPosition = Double.NEGATIVE_INFINITY;
  }
}