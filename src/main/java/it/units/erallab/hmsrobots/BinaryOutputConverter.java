package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.rl.DiscreteRL;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.ArrayList;
import java.util.function.Function;

import static java.lang.Math.abs;

class BinaryOutputConverter implements DiscreteRL.OutputConverter {
  private final int outputDimension;
  private final int numberClusters;

  private final double force;

  private final ArrayList<ArrayList<Grid.Key>> clusters;


  BinaryOutputConverter(
      int outputDimension,
      ArrayList<ArrayList<Grid.Key>> clusters,
      double force
  ) {
    this.outputDimension = outputDimension;
    this.clusters = clusters;

    this.numberClusters = clusters.size();
    this.force = abs(force);
  }

  BinaryOutputConverter(
      int outputDimension,
      ArrayList<ArrayList<Grid.Key>> clusters
  ) {
    this(outputDimension, clusters, 1.0);
  }

  @Override
  public double[] apply(Integer integer) {
    double[] controls = new double[numberClusters];
    // TODO Check for improvements
    char[] binary = String.format("%04d", Integer.parseInt(Integer.toBinaryString(integer))).toCharArray();
    for (int i = 0; i < numberClusters; i++) {
      controls[i] = (binary[i] == '0') ? force : -force;
    }

    double[] output = new double[outputDimension];
    int j = 0;
    for (int i = 0; i < numberClusters; i++) {
      for (Grid.Key ignored : clusters.get(i)) {
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
    return 1;
  }

  @Override
  public int getOutputDimension() {
    return outputDimension;
  }
}
