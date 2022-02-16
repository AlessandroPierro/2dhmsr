package it.units.erallab.hmsrobots.core.snapshots;

public class QTableState {

  private final double[][] qTable;
  private final int statesNumber;
  private final int actionsNumber;
  private final double learningRate;
  private final double explorationRate;
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final double discountFactor;


  public QTableState(
      double[][] qTable,
      int statesNumber,
      int actionsNumber,
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor
  ) {
    this.qTable = qTable;
    this.statesNumber = statesNumber;
    this.actionsNumber = actionsNumber;
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
  }

  public int getActionsNumber() {
    return actionsNumber;
  }

  public double getDiscountFactor() {
    return discountFactor;
  }

  public double getExplorationRate() {
    return explorationRate;
  }

  public double getExplorationRateDecay() {
    return explorationRateDecay;
  }

  public double getLearningRate() {
    return learningRate;
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
