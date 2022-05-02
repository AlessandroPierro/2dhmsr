package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.geometry.Point2;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.Objects;
import java.util.function.Function;

public class RLController extends AbstractController implements Snapshottable, Serializable {

    @JsonProperty
    private final ClusteredObservationFunction observationFunction;
    @JsonProperty
    private RewardFunction rewardFunction;
    @JsonProperty
    private final ContinuousRL rl;
    @JsonProperty
    private final Function<double[], Grid<Double>> controlFunction;

    private double currentVelocity;
    private double[] observation;
    private double reward;
    private double[] action;

    @JsonCreator
    public RLController(
            @JsonProperty("observationFunction") ClusteredObservationFunction observationFunction,
            @JsonProperty("rewardFunction") RewardFunction rewardFunction,
            @JsonProperty("rl") ContinuousRL rl,
            @JsonProperty("controlFunction") Function<double[], Grid<Double>> controlFunction
    ) {
        this.observationFunction = observationFunction;
        this.rewardFunction = rewardFunction;
        this.rl = rl;
        this.controlFunction = controlFunction;
    }

    @Override
    public Grid<Double> computeControlSignals(
            double t, Grid<Voxel> voxels
    ) {
        // Track the current velocity
        int nVoxels = (int) voxels.values().stream().filter(Objects::nonNull).count();
        currentVelocity = voxels.values().stream().filter(Objects::nonNull).map(Voxel::getLinearVelocity).map(Point2::x).reduce(0.0, Double::sum) / nVoxels;

        observation = observationFunction.apply(t, voxels);
        reward = rewardFunction.apply(voxels);
        action = rl.apply(t, observation, reward);
        return controlFunction.apply(action);
    }

    @Override
    public Snapshot getSnapshot() {
        Snapshot snapshot = new Snapshot(new RLControllerState(reward, observation, action, currentVelocity), getClass());
        snapshot.getChildren().add(rl.getSnapshot());
        return snapshot;
    }

    @Override
    public void reset() {
        rl.reset();
        rewardFunction.reset();
        if (observationFunction instanceof Resettable r) {
            r.reset();
        }
        if (controlFunction instanceof Resettable r) {
            r.reset();
        }
    }

    public void setRewardFunction(RewardFunction rewardFunction) {
        this.rewardFunction = rewardFunction;
    }

    public ContinuousRL getRL() {
        return rl;
    }
}