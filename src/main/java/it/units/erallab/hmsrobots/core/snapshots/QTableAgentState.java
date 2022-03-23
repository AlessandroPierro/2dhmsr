package it.units.erallab.hmsrobots.core.snapshots;

import java.util.Arrays;

public class QTableAgentState {

  private final double[][] qTable;
  private final int statesNumber;
  private final int actionsNumber;
  private final double learningRateDecay;
  private final double discountFactor;
  private final int state;


  public QTableAgentState(
      double[][] qTable,
      int statesNumber,
      int actionsNumber,
      double learningRateDecay,
      double discountFactor,
      int state
  ) {
    this.qTable = copyOf(qTable);
    this.statesNumber = statesNumber;
    this.actionsNumber = actionsNumber;
    this.learningRateDecay = learningRateDecay;
    this.discountFactor = discountFactor;
    this.state = state;
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

  public double getLearningRateDecay() {
    return learningRateDecay;
  }

  public int getStatesNumber() {
    return statesNumber;
  }

  public double[][] getqTable() {
    return qTable;
  }

  public int getState() {
    return state;
  }
}
