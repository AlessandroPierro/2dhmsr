package it.units.erallab.hmsrobots.core.snapshots;

import java.util.Arrays;

public class QTableAgentState {

  private final double[][] qTable;
  private final int statesNumber;
  private final int actionsNumber;
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final double discountFactor;


  public QTableAgentState(
      double[][] qTable,
      int statesNumber,
      int actionsNumber,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor
  ) {
    this.qTable = copyOf(qTable);
    this.statesNumber = statesNumber;
    this.actionsNumber = actionsNumber;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
  }

  private static double[][] copyOf(double[][] o) {
    double[][] c = new double[o.length][];
    for (int i = 0; i < o.length; i++) {
      c[i] = Arrays.copyOf(o[i], o[i].length);
    }
    return c;
  }

  public int getActionsNumber() {
    return actionsNumber;
  }

  public double getDiscountFactor() {
    return discountFactor;
  }

  public double getExplorationRateDecay() {
    return explorationRateDecay;
  }

  public double getLearningRateDecay() {
    return learningRateDecay;
  }

  public int getStatesNumber() {
    return statesNumber;
  }

  public double[][] getqTable() {
    return qTable;
  }
}
