package it.units.erallab.hmsrobots.core.snapshots;

public class QTableAgentState {

  private final double[][] qTable;
  private final int stateDim;
  private final int actionDim;
  private final double learningRate;
  private final double explorationRate;
  private final int currentState;
  private final int currentAction;
  private final double currentReward;


  public QTableAgentState(
      double[][] qTable,
      int stateDim,
      int actionDim,
      double learningRate,
      double explorationRate,
      int currentState,
      int currentAction,
      double currentReward
  ) {
    this.qTable = qTable;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.currentState = currentState;
    this.currentAction = currentAction;
    this.currentReward = currentReward;
  }

  public double[][] getqTable() {
    return qTable;
  }

  public int getStateDim() {
    return stateDim;
  }

  public int getActionDim() {
    return actionDim;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public double getExplorationRate() {
    return explorationRate;
  }

  public int getCurrentState() {
    return currentState;
  }

  public int getCurrentAction() {
    return currentAction;
  }

  public double getCurrentReward() {
    return currentReward;
  }
}
