/*
 * Copyright (C) 2021 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package it.units.erallab.hmsrobots.core.controllers;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class PruningMLP extends MultiLayerPerceptron {

  public enum Context {WHOLE, LAYER, NEURON}

  public enum Criterion {
    WEIGHT,
    SIGNAL_MEAN,
    ABS_SIGNAL_MEAN,
    SIGNAL_VARIANCE
  }

  @JsonProperty
  private final long nOfCalls;
  @JsonProperty
  private final Context context;
  @JsonProperty
  private final Criterion criterion;
  @JsonProperty
  private final double rate;

  private long counter;

  private double[][][] means;
  private double[][][] absMeans;
  private double[][][] meanDiffSquareSums; //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm

  public PruningMLP(
      @JsonProperty("activationFunction") ActivationFunction activationFunction,
      @JsonProperty("weights") double[][][] weights,
      @JsonProperty("neurons") int[] neurons,
      @JsonProperty("nOfCalls") long nOfCalls,
      @JsonProperty("context") Context context,
      @JsonProperty("criterion") Criterion criterion,
      @JsonProperty("rate") double rate
  ) {
    super(activationFunction, weights, neurons);
    this.nOfCalls = nOfCalls;
    this.context = context;
    this.criterion = criterion;
    this.rate = rate;
    reset();
  }

  public PruningMLP(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, double[] weights, long nOfCalls, Context context, Criterion criterion, double rate) {
    super(activationFunction, nOfInput, innerNeurons, nOfOutput, weights);
    this.nOfCalls = nOfCalls;
    this.context = context;
    this.criterion = criterion;
    this.rate = rate;
    reset();
  }

  public PruningMLP(ActivationFunction activationFunction, int nOfInput, int[] innerNeurons, int nOfOutput, long nOfCalls, Context context, Criterion criterion, double rate) {
    super(activationFunction, nOfInput, innerNeurons, nOfOutput);
    this.nOfCalls = nOfCalls;
    this.context = context;
    this.criterion = criterion;
    this.rate = rate;
    reset();
  }

  @Override
  public void setParams(double[] params) {
    super.setParams(params);
    reset();
  }

  private void reset() {
    if (rate < 0 || rate > 1) {
      throw new IllegalArgumentException(String.format("Pruning rate should be defined in [0,1]: %f found", rate));
    }
    counter = 0;
    means = new double[weights.length][][];
    absMeans = new double[weights.length][][];
    meanDiffSquareSums = new double[weights.length][][];
    for (int i = 1; i < neurons.length; i++) {
      means[i - 1] = new double[weights[i - 1].length][];
      absMeans[i - 1] = new double[weights[i - 1].length][];
      meanDiffSquareSums[i - 1] = new double[weights[i - 1].length][];
      for (int j = 0; j < weights[i - 1].length; j++) {
        means[i - 1][j] = new double[weights[i - 1][j].length];
        absMeans[i - 1][j] = new double[weights[i - 1][j].length];
        meanDiffSquareSums[i - 1][j] = new double[weights[i - 1][j].length];
        for (int k = 0; k < weights[i - 1][j].length; k++) {
          means[i - 1][j][k] = weights[i - 1][j][k];
          absMeans[i - 1][j][k] = Math.abs(weights[i - 1][j][k]);
        }
      }
    }
  }

  private void prune() {
    List<Pair<int[], Double>> pairs = new ArrayList<>();
    for (int i = 1; i < neurons.length; i++) {
      for (int j = 0; j < neurons[i]; j++) {
        for (int k = 0; k < neurons[i - 1] + 1; k++) {
          pairs.add(Pair.create(
              new int[]{i, j, k},
              switch (criterion) {
                case WEIGHT -> Math.abs(weights[i - 1][j][k]);
                case SIGNAL_MEAN -> means[i - 1][j][k];
                case ABS_SIGNAL_MEAN -> absMeans[i - 1][j][k];
                case SIGNAL_VARIANCE -> meanDiffSquareSums[i - 1][j][k];
              }
          ));
        }
      }
    }
    if (context.equals(Context.WHOLE)) {
      prune(pairs);
    } else if (context.equals(Context.LAYER)) {
      for (int i = 1; i < neurons.length; i++) {
        final int localI = i;
        prune(pairs.stream().filter(p -> p.getKey()[0] == localI).collect(Collectors.toList()));
      }
    } else if (context.equals(Context.NEURON)) {
      for (int i = 1; i < neurons.length; i++) {
        for (int j = 0; j < neurons[i]; j++) {
          final int localI = i;
          final int localJ = j;
          prune(pairs.stream().filter(p -> p.getKey()[0] == localI && p.getKey()[1] == localJ).collect(Collectors.toList()));
        }
      }
    }
  }

  private void prune(List<Pair<int[], Double>> localPairs) {
    localPairs.sort(Comparator.comparing(Pair::getValue));
    localPairs.subList(0, (int) Math.round(localPairs.size() * rate)).forEach(p -> prune(p.getKey()));
  }

  private void prune(int[] is) {
    int i = is[0];
    int j = is[1];
    int k = is[2];
    weights[i - 1][j][k] = 0d;
    if (criterion.equals(Criterion.SIGNAL_VARIANCE) && k != 0) {
      weights[i - 1][j][0] = weights[i - 1][j][0] + means[i - 1][j][k];
    }
  }

  @Override
  public double[] apply(double[] input) {
    if (nOfCalls == counter) {
      prune();
    }
    if (input.length != neurons[0]) {
      throw new IllegalArgumentException(String.format("Expected input length is %d: found %d", neurons[0], input.length));
    }
    double[][] values = new double[neurons.length][];
    values[0] = new double[neurons[0]];
    System.arraycopy(input, 0, values[0], 0, input.length);
    for (int i = 1; i < neurons.length; i++) {
      values[i] = new double[neurons[i]];
      for (int j = 0; j < neurons[i]; j++) {
        double sum = weights[i - 1][j][0]; //set the bias
        for (int k = 1; k < neurons[i - 1] + 1; k++) {
          double signal = values[i - 1][k - 1] * weights[i - 1][j][k];
          sum = sum + signal;
          double delta = signal - means[i - 1][j][k];
          means[i - 1][j][k] = means[i - 1][j][k] + delta / ((double) counter + 1d);
          absMeans[i - 1][j][k] = absMeans[i - 1][j][k] + (Math.abs(signal) - absMeans[i - 1][j][k]) / ((double) counter + 1d);
          meanDiffSquareSums[i - 1][j][k] = meanDiffSquareSums[i - 1][j][k] + delta * (signal - means[i - 1][j][k]);
        }
        values[i][j] = activationFunction.getF().apply(sum);
      }
    }
    counter = counter + 1;
    return values[neurons.length - 1];
  }
}
