package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;

import java.util.Arrays;

public class RLListener implements SnapshotListener {

  double lastT = 0d;

  RLControllerState extractState(Snapshot snapshot) {
    if (snapshot.getContent() instanceof RLControllerState state) {
      return state;
    } else {
      for (Snapshot child : snapshot.getChildren()) {
        RLControllerState state = extractState(child);
        if (state != null) {
          return state;
        }
      }
    }
    return null;
  }

  @Override
  public void listen(double t, Snapshot snapshot) {
    RLControllerState state = extractState(snapshot);
    System.out.println(state.getReward()+"\t"+ Arrays.toString(state.getAction()) +"\t"+ Arrays.toString(state.getObservation()));
  }
}
