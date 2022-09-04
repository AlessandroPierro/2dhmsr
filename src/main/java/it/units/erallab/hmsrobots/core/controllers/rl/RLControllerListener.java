package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RLControllerListener implements SnapshotListener {

  private final List<RLEvent> history;
  private double t0 = 0d;

  public RLControllerListener() {
    this.history = new ArrayList<>();
  }

  record RLEvent(double time, double reward, double velocityX, double velocityY) {}

  private static File check(File file) {
    String originalFileName = file.getPath();
    while (file.exists()) {
      String newName = null;
      Matcher mNum = Pattern.compile("\\((?<n>[0-9]+)\\)\\.\\w+$").matcher(file.getPath());
      if (mNum.find()) {
        int n = Integer.parseInt(mNum.group("n"));
        newName = new StringBuilder(file.getPath()).replace(mNum.start("n"), mNum.end("n"), Integer.toString(n + 1))
            .toString();
      }
      Matcher mExtension = Pattern.compile("\\.\\w+$").matcher(file.getPath());
      if (newName == null && mExtension.find()) {
        newName = new StringBuilder(file.getPath()).replace(
            mExtension.start(),
            mExtension.end(),
            ".(1)" + mExtension.group()
        ).toString();
      }
      if (newName == null) {
        newName = file.getPath() + ".newer";
      }
      file = new File(newName);
    }
    return file;
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
    if (t - t0 >= 0.25 || t - t0 < 0) {
      RLControllerState controllerState = extractControllerState(snapshot);
      RLEvent event = new RLEvent(t, controllerState.reward(), 0d, 0d);
      history.add(event);
      t0 = t;
    }
  }

  public void toFile(File file) {
    List<String> lines = history.stream()
        .map(event -> String.format("%f;%f;%f;%f", event.time, event.reward, event.velocityX, event.velocityY))
        .map(s -> s.replace(",", "."))
        .map(s -> s.replace(";", ","))
        .toList();
    file = check(file);
    try {
      FileWriter fileWriter = new FileWriter(file);
      fileWriter.write("time,reward,velocityX,velocityY" + System.lineSeparator());
      for (String line : lines) {
        fileWriter.write(line + System.lineSeparator());
      }
      fileWriter.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public List<Double> extractRewards() {
    return history.stream().map(event -> event.reward).toList();
  }

  public List<Double> extractVelocities() {
    return history.stream().map(event -> event.velocityX).toList();
  }

}
