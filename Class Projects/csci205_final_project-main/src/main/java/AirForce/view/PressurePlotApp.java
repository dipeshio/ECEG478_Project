/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Jack Mclaud
 * Date: 4/24/25
 * Time: 13:34
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: PressurePlotApp
 *
 * Description:
 * Manages the pressure data for the dashboard's ProgressBar and Label with animated progress updates.
 *
 * ****************************************
 */

package AirForce.view;

import javafx.scene.control.TextField;
import javafx.scene.layout.Pane;
import javafx.scene.shape.Arc;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * A utility class for visualizing pressure data in the AirForce application.
 * Manages a pressure gauge by updating a text label and rotating a needle based on pressure values
 * from a data queue. Integrates with JavaFX components for UI display.
 */
public class PressurePlotApp {
    private final ConcurrentLinkedQueue<Number> pressureDataQ; // Queue for pressure data
    private final TextField pressureLabel;                     // Label to display pressure value
    private static final double MAX_PRESSURE = 120000.0;       // Maximum pressure in Pascals (120 kPa)
    private Arc needle;                                        // Needle for the pressure gauge

    /**
     * Constructs a PressurePlotApp instance with the specified label, gauge pane, and data queue.
     * Initializes the pressure label to display zero pressure.
     *
     * @param pressureLabel the {@link TextField} to display the pressure value
     * @param gaugePane    the {@link Pane} containing the pressure gauge visualization
     * @param pressureDataQ the {@link ConcurrentLinkedQueue} containing pressure data
     */
    public PressurePlotApp(TextField pressureLabel, Pane gaugePane, ConcurrentLinkedQueue<Number> pressureDataQ) {
        this.pressureLabel = pressureLabel;
        this.pressureDataQ = pressureDataQ;
        initialize();
    }

    /**
     * Initializes the pressure label with a default value of zero pressure.
     */
    private void initialize() {
        if (pressureLabel != null) {
            pressureLabel.setText("Pressure: 0 kPa");
        }
    }

    /**
     * Sets the needle for the pressure gauge and initializes its position to the middle of the green zone.
     *
     * @param needle the {@link Arc} representing the gauge needle
     */
    public void setNeedle(Arc needle) {
        this.needle = needle;
        // Initialize needle to green zone (middle)
        needle.setStartAngle(90.0); // 90° corresponds to the middle of green (-60° to 0°)
    }


    /**
     * Updates the pressure label and gauge needle based on the latest pressure data.
     * If the BMP sensor is off or no data is available, updates the label accordingly.
     *
     * @param isBMPOn indicates if the BMP180 sensor is enabled
     */
    public void addDataToSeries(boolean isBMPOn) {
        if (isBMPOn && !pressureDataQ.isEmpty()) {
            double pressurePa = pressureDataQ.remove().doubleValue();
            double pressureKPa = pressurePa / 1000.0;
            if (pressureLabel != null) {
                pressureLabel.setText(String.format("Pressure: %.1f kPa", pressureKPa));
            }

            // Set needle statically based on pressure value
            if (pressureKPa <= 1.0) {
                // Green zone (0-1 kPa, -60° to 0°), stay at 90° (middle of green)
                needle.setStartAngle(90.0);
            } else {
                // Yellow zone (>1 kPa, 0° to 120°), scale from 1 kPa to 120 kPa
                double normalizedValue = (pressureKPa - 1.0) / (MAX_PRESSURE / 1000.0 - 1.0); // 0 to 1 for 1 to 120 kPa
                if (normalizedValue > 1.0) normalizedValue = 1.0;
                double angle = 90.0 - (normalizedValue * 120.0); // 90° to 0° (0° to 120° on gauge)
                needle.setStartAngle(angle);
            }

        } else {
            if (!isBMPOn && pressureLabel != null) {
                pressureLabel.setText("Pressure: OFF");
            } else if (pressureDataQ.isEmpty() && pressureLabel != null) {
                pressureLabel.setText("Pressure: No data");
            }
        }
    }


    /**
     * Returns the queue containing pressure data.
     *
     * @return the {@link ConcurrentLinkedQueue} of pressure data
     */
    public ConcurrentLinkedQueue<Number> getPressureDataQ() {
        return pressureDataQ;
    }
}