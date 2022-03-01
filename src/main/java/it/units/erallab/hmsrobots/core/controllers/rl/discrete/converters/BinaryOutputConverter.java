package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.function.Function;

import static java.lang.Math.abs;

public class BinaryOutputConverter implements DiscreteRL.OutputConverter {
  private final int outputDimension;
  private final int numberClusters;

  private final double force;

  private final List<List<Grid.Key>> clusters;


  public BinaryOutputConverter(
      int outputDimension,
      List<List<Grid.Key>> clusters,
      double force
  ) {
    this.outputDimension = outputDimension;
    this.clusters = clusters;

    this.numberClusters = clusters.size();
    this.force = abs(force);
  }

  BinaryOutputConverter(
      int outputDimension,
      List<List<Grid.Key>> clusters
  ) {
    this(outputDimension, clusters, 1.0);
  }

  @Override
  public double[] apply(Integer integer) {
    double[] controls = new double[numberClusters];
    char[] binary = String.format("%0"+numberClusters+"d", Integer.parseInt(Integer.toBinaryString(integer))).toCharArray();
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
