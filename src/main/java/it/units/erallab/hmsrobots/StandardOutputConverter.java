package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.rl.DiscreteRL;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.function.Function;

import static java.lang.Math.random;

class StandardOutputConverter implements DiscreteRL.OutputConverter {
  private final int inputDimension = 1;
  private final int outputDimension;
  private final int numberClusters;

  private final Grid<Voxel> body;
  private final ArrayList<ArrayList<Grid.Key>> clusters;


  StandardOutputConverter(
      int outputDimension,
      Grid<Voxel> body,
      ArrayList<ArrayList<Grid.Key>> clusters
  ) {
    this.outputDimension = outputDimension;
    this.body = body;
    this.clusters = clusters;

    this.numberClusters = clusters.size();
  }

  @Override
  public double[] apply(Integer integer) {
    double[] controls = new double[numberClusters];
    for (int i = 0; i < numberClusters; i++) {
      controls[i] = (integer % (i+2)) == 0 ? 1.0 : -1.0;
    }

    double[] output = new double[outputDimension];
    int j = 0;
    for (int i = 0; i < numberClusters; i++) {
      for (Grid.Key key : clusters.get(i)) {
        output[j] = controls[i];
        j++;
      }
    }
    return output;
  }

  @Override
  public <V> Function<V, double[]> compose(Function<? super V, ? extends Integer> before) {
    return DiscreteRL.OutputConverter.super.compose(before);
  }

  @Override
  public <V> Function<Integer, V> andThen(Function<? super double[], ? extends V> after) {
    return DiscreteRL.OutputConverter.super.andThen(after);
  }

  @Override
  public int getInputDimension() {
    return inputDimension;
  }

  @Override
  public int getOutputDimension() {
    return outputDimension;
  }
}
