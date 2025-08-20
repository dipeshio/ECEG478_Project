/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 5/2/25
 * Time: 11:53
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: temperaturePlotApp
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.view;

import javafx.scene.control.TextField;
import javafx.scene.layout.Pane;
import javafx.scene.shape.Arc;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * A utility class for visualizing temperature data in the AirForce application.
 * Manages a temperature gauge by updating a text label and rotating a needle based on temperature values
 * from a data queue. Integrates with JavaFX components for UI display.
 */
public class TemperaturePlotApp {
    private final ConcurrentLinkedQueue<Number> tempDataQ; // Queue for temperature data
    private final TextField tempLabel;                     // Label to display temperature value
    private static final double MAX_TEMP = 50.0;           // Maximum temperature in Celsius
    private Arc needle;                                    // Needle for the temperature gauge

    /**
     * Constructs a TemperaturePlotApp instance with the specified label, gauge pane, and data queue.
     * Initializes the temperature label to display zero temperature.
     *
     * @param tempLabel    the {@link TextField} to display the temperature value
     * @param gaugePane    the {@link Pane} containing the temperature gauge visualization
     * @param tempDataQ    the {@link ConcurrentLinkedQueue} containing temperature data
     */
    public TemperaturePlotApp(TextField tempLabel, Pane gaugePane, ConcurrentLinkedQueue<Number> tempDataQ) {
        this.tempLabel = tempLabel;
        this.tempDataQ = tempDataQ;
        initialize();
    }

    /**
     * Initializes the temperature label with a default value of zero temperature.
     */
    private void initialize() {
        if (tempLabel != null) {
            tempLabel.setText("Temp: 0 C");
        }
    }

    /**
     * Sets the needle for the temperature gauge and initializes its position to the middle of the green zone.
     *
     * @param needle the {@link Arc} representing the gauge needle
     */
    public void setNeedle(Arc needle) {
        this.needle = needle;
        // Initialize needle to green zone
        needle.setStartAngle(0.0); // 90° corresponds to the middle of green (-60° to 0°)
    }

    /**
     * Updates the temperature label and gauge needle based on the latest temperature data.
     * If the BMP sensor is off or no data is available, updates the label accordingly.
     *
     * @param isBMPOn indicates if the BMP180 sensor is enabled
     */
    public void addDataToSeries(boolean isBMPOn) {
        if (isBMPOn && !tempDataQ.isEmpty()) {
            double tempC = tempDataQ.remove().doubleValue();
            if (tempLabel != null) {
                tempLabel.setText(String.format("Temp: %.0f C", tempC));
            }

            // Set needle statically based on temperature value
            if (tempC <= 1.0) {
                // Green zone (0-1°C, -60° to 0°), stay at 90° (middle of green)
                needle.setStartAngle(90.0);
            } else {
                // Yellow zone (>1°C, 0° to 120°), scale from 1°C to 50°C
                double normalizedValue = (tempC - 1.0) / (MAX_TEMP - 1.0); // 0 to 1 for 1 to 50°C
                if (normalizedValue > 1.0) normalizedValue = 1.0;
                double angle = 90.0 - (normalizedValue * 120.0); // 90° to 0° (0° to 120° on gauge)
                needle.setStartAngle(angle);
            }
        } else {
            if (!isBMPOn && tempLabel != null) {
                tempLabel.setText("Temp: OFF");
            } else if (tempDataQ.isEmpty() && tempLabel != null) {
                tempLabel.setText("Temp: No data");
            }
        }
    }
}