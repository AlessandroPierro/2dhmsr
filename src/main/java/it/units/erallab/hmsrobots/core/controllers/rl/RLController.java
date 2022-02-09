package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.ToDoubleFunction;

public class RLController extends AbstractController {

  private final ToDoubleFunction<Grid<Voxel>> rewardFunction;
  private final BiFunction<Double, Grid<Voxel>, double[]> observationFunction;
  private final ContinuousRL rl;
  private final int frequency;
  private final ArrayList<ArrayList<Grid.Key>> clusters;
  private int step = 0;
  private Grid<Double> output;
  private boolean initialized = false;

  public RLController(
      ToDoubleFunction<Grid<Voxel>> rewardFunction,
      BiFunction<Double, Grid<Voxel>, double[]> observationFunction,
      ContinuousRL rl,
      int frequency,
      ArrayList<ArrayList<Grid.Key>> clusters
  ) {
    this.rewardFunction = rewardFunction;
    this.observationFunction = observationFunction;
    this.rl = rl;
    this.frequency = frequency;
    this.clusters = clusters;
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    if (!initialized) {
      output = Grid.create(voxels.getW(), voxels.getH());
      initialized = true;
    }
    if ((step % frequency) == 0) {
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
      int c = 0;
      for (ArrayList<Grid.Key> cluster : clusters) {
        for (Grid.Key key : cluster) {
          output.set(key.x(), key.y(), action[c]);
          c++;
        }
      }
    }
    step = (step % frequency) + 1;
    return output;
  }

  @Override
  public void reset() {
    if (rl instanceof Resettable r) {
      r.reset();
    }
  }
}
