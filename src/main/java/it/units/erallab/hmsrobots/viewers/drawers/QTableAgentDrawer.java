package it.units.erallab.hmsrobots.viewers.drawers;

import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.viewers.DrawingUtils;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.SortedMap;

public class QTableAgentDrawer extends MemoryDrawer<QTableAgentState> {

  private final Color minColor;
  private final Color maxColor;

  public QTableAgentDrawer(Extractor extractor, double windowT, Color minColor, Color maxColor) {
    super(extractor, s -> (QTableAgentState) s.getContent(), windowT);
    this.minColor = minColor;
    this.maxColor = maxColor;
  }

  public QTableAgentDrawer(Extractor extractor, double windowT) {
    this(extractor, windowT, DrawingUtils.Colors.DATA_ZERO, DrawingUtils.Colors.DATA_NEGATIVE);
  }

  private void drawHeatmap(double[][] qTable, BoundingBox bb, Graphics2D g) {
    double min = -2; // Arrays.stream(qTable).flatMapToDouble(Arrays::stream).min().orElse(Double.NEGATIVE_INFINITY);
    double max = 2; // Arrays.stream(qTable).flatMapToDouble(Arrays::stream).max().orElse(Double.POSITIVE_INFINITY);

    int nStates = qTable.length;
    int nActions = qTable[0].length;
    double cellWidth = bb.width() / nStates;
    double cellHeight = bb.height() / nActions;
    for (int x = 0; x < nStates; x++) {
      for (int y = 0; y < nActions; y++) {
        Rectangle2D cell = new Rectangle2D.Double(
            bb.min().x() + x * cellWidth,
            bb.min().y() + y * cellHeight,
            cellWidth,
            cellHeight
        );
        g.setColor(DrawingUtils.linear(minColor, maxColor, (float) min, (float) max, (float) qTable[x][y]));
        g.fill(cell);
      }
    }
  }

  @Override
  protected void innerDraw(double t, Snapshot snapshot, SortedMap<Double, QTableAgentState> memory, Graphics2D g) {
    QTableAgentState current = memory.get(memory.lastKey());
    BoundingBox bb = BoundingBox.of(
        g.getClip().getBounds2D().getMinX(),
        g.getClip().getBounds2D().getY(),
        g.getClip().getBounds2D().getMaxX(),
        g.getClip().getBounds2D().getMaxY()
    );

    drawHeatmap(current.getqTable(), bb, g);

  }

}