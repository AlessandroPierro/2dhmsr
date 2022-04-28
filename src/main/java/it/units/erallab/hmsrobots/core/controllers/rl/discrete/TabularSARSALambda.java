package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import org.apache.commons.math3.linear.RealMatrix;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

public class TabularSARSALambda extends AbstractQTableAgent {

    private final double lambda;
    private RealMatrix eTraces;

    public TabularSARSALambda(double discountFactor, double lambda, int stateDim, int actionDim, double meanQ, double stdQ, int seed) {
        super(discountFactor, stateDim, actionDim, meanQ, stdQ, seed);
        this.lambda = lambda;
        this.eTraces = createRealMatrix(stateDim, actionDim);
    }

    @Override
    public void reset() {
        super.reset();
        eTraces = createRealMatrix(stateDim, actionDim);
    }

    @Override
    protected void updateQTable(int newState, double reward, int newAction) {
        double delta = reward + discountFactor * qTable.getEntry(newState, newAction) - qTable.getEntry(previousState, previousAction);
        eTraces.addToEntry(previousState, previousAction, 1);
        qTable = qTable.add(eTraces.scalarMultiply(lambda).scalarMultiply(delta));
        eTraces = eTraces.scalarMultiply(lambda).scalarMultiply(discountFactor);
    }

}
