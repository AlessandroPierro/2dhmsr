package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.ToDoubleFunction;

public class RLController extends AbstractController {

  private final ToDoubleFunction<Grid<Voxel>> rewardFunction;
  private final BiFunction<Double, Grid<Voxel>, double[]> observationFunction;
  private final ContinuousRL rl;

  public RLController(
      ToDoubleFunction<Grid<Voxel>> rewardFunction,
      BiFunction<Double, Grid<Voxel>, double[]> observationFunction,
      ContinuousRL rl
  ) {
    this.rewardFunction = rewardFunction;
    this.observationFunction = observationFunction;
    this.rl = rl;
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    double[] observation = observationFunction.apply(t, voxels);
    if (observation.length != rl.getInputDimension()) {
      throw new IllegalArgumentException(String.format(
          "Observation dim different than expected: %d vs %d",
          observation.length,
          rl.getInputDimension()
      ));
    }
    double reward = rewardFunction.applyAsDouble(voxels);
    double[] action = rl.apply(t, observation, reward);
    int nOfVoxels = (int) voxels.stream().map(Grid.Entry::value).filter(Objects::nonNull).count();
    if (action.length != nOfVoxels) {
      throw new IllegalArgumentException(String.format(
          "Action dim different than expected: %d vs %d",
          action.length,
          nOfVoxels
      ));
    }
    Grid<Double> output = Grid.create(voxels.getW(), voxels.getH());
    int c = 0;
    for (Grid.Entry<Voxel> e : voxels) {
      if (e.value() != null) {
        output.set(e.key().x(), e.key().y(), action[c]);
        c = c + 1;
      }
    }
    return output;
  }

  @Override
  public void reset() {
    if (rl instanceof Resettable r) {
      r.reset();
    }
  }
}
