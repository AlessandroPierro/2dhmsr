package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;

import java.util.Arrays;

public class RLListener implements SnapshotListener {

  double lastT = 0d;

  QTableAgentState extractAgentState(Snapshot snapshot) {
    if (snapshot.getContent() instanceof QTableAgentState state) {
      return state;
    } else {
      for (Snapshot child : snapshot.getChildren()) {
        QTableAgentState state = extractAgentState(child);
        if (state != null) {
          return state;
        }
      }
    }
    return null;
  }

  RLControllerState extractControllerState(Snapshot snapshot) {
    if (snapshot.getContent() instanceof RLControllerState state) {
      return state;
    } else {
      for (Snapshot child : snapshot.getChildren()) {
        RLControllerState state = extractControllerState(child);
        if (state != null) {
          return state;
        }
      }
    }
    return null;
  }

  @Override
  public void listen(double t, Snapshot snapshot) {
    RLControllerState controllerState = extractControllerState(snapshot);
    QTableAgentState rlState = extractAgentState(snapshot);
    System.out.println(controllerState.getReward() + "\t"
        + Arrays.toString(controllerState.getAction()) + "\t"
        + rlState.getState() + "\t"
        + Arrays.toString(controllerState.getObservation()));
  }
}
