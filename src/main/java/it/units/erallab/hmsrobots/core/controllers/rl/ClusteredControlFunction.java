package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.IOSized;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.function.Function;

public class ClusteredControlFunction implements Function<double[], Grid<Double>>, IOSized {
  private final List<List<Grid.Key>> clusters;
  private final int nClusters;
  private final int nVoxels;
  private final int w;
  private final int h;

  public ClusteredControlFunction(List<List<Grid.Key>> clusters) {
    this.clusters = clusters;
    this.nClusters = clusters.size();
    this.nVoxels = clusters.stream().mapToInt(List::size).sum();
    this.w = clusters.stream()
        .mapToInt(c -> c.stream().mapToInt(Grid.Key::x).max().orElse(0))
        .max().orElse(0) + 1;
    this.h = clusters.stream()
        .mapToInt(c -> c.stream().mapToInt(Grid.Key::y).max().orElse(0))
        .max().orElse(0) + 1;
  }

  @Override
  public Grid<Double> apply(double[] doubles) {
    Grid<Double> control = Grid.create(w, h);
    for (int i = 0; i < nClusters; i++) {
      for (Grid.Key key : clusters.get(i)) {
        control.set(key.x(), key.y(), doubles[i]);
      }
    }
    return control;
  }

  @Override
  public int getInputDimension() {
    return nClusters;
  }

  @Override
  public int getOutputDimension() {
    return nVoxels;
  }
}
