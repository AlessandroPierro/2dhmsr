package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import java.util.function.Supplier;

public class DoubleQLearningAgent extends AbstractQTableAgent {
  private double[][] qTableB;
  private boolean lastTableIsA;

  public DoubleQLearningAgent(
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      Supplier<Double> initializer,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    super(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
        initializer,
        stateSpaceDimension,
        actionSpaceDimension
    );

    this.qTableB = new double[stateSpaceDimension][actionSpaceDimension];
    for (int i = 0; i < stateSpaceDimension; i++) {
      for (int j = 0; j < actionSpaceDimension; j++) {
        this.qTableB[i][j] = initializer.get();
      }
    }
  }

  @Override
  public int apply(double t, int newState, double reward) {
    if (initialized) {
      if (random.nextDouble() < 0.5) {
        updateQTable(previousState, action, reward, newState, qTableA, qTableB);
      } else {
        updateQTable(previousState, action, reward, newState, qTableB, qTableA);
      }
    } else {
      initialized = true;
    }

    updateRates(newState);

    action = random.nextDouble() < explorationRates[newState] ? random.nextInt(actionSpaceDimension) : getMaxAction(
        newState,
        qTableA,
        qTableB
    );
    previousState = newState;
    return action;
  }

  void updateQTable(int previousState, int action, double reward, int newState, double[][] qTableA, double[][] qTableB) {
    int maxAction = getMaxAction(previousState, qTableA);
    qTableA[previousState][action] = qTableA[previousState][action] + learningRates[previousState] * (
        reward + discountFactor * qTableB[newState][maxAction] - qTableA[previousState][action]
    );
  }

  int getMaxAction(int state, double[][] qTableA, double[][] qTableB) {
    int maxAction = 0;
    double maxQ = qTableA[state][0] + qTableB[state][0];
    for (int i = 1; i < actionSpaceDimension; i++) {
      double tempQ = qTableA[state][i] + qTableB[state][i];
      if (tempQ > maxQ) {
        maxQ = tempQ;
        maxAction = i;
      }
    }
    return maxAction;
  }
}
