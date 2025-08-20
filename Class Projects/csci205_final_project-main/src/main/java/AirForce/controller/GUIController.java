/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/16/2025
 * Time: 3:27 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.controller
 * Class: GUIController
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.controller;

import AirForce.model.CountDownCheck;
import AirForce.model.SystemOperation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.Label;
import javafx.scene.control.ToggleButton;
import javafx.util.Duration;

/**
 * Controller class for managing system operations and countdown timer in the AirForce application.
 * Interfaces with {@link SystemOperation} to control power, sensors, telemetry, GPS, and recording,
 * and with {@link CountDownCheck} to manage the countdown timer. Uses JavaFX for UI interactions.
 */
public class GUIController {
    private final SystemOperation systemOperation; // Manages system operations
    private final CountDownCheck countDownCheck;   // Manages countdown timer
    private Timeline countdownTimeline;            // Timeline for countdown updates

    /**
     * Constructs a GUIController with the specified system operation and countdown check.
     *
     * @param so  the {@link SystemOperation} instance for controlling system functions
     * @param cdc the {@link CountDownCheck} instance for managing the countdown timer
     */
    public GUIController(SystemOperation so, CountDownCheck cdc) {
        this.systemOperation = so;
        this.countDownCheck = cdc;
    }

    /**
     * Starts the countdown timer and updates the provided label with the remaining time.
     * The timer updates every second and stops when the countdown is finished.
     *
     * @param countdownLabel the {@link Label} to display the formatted remaining time
     */
    public void startCountdownTimer(Label countdownLabel) {
        countDownCheck.start();
        countdownTimeline = new Timeline(new KeyFrame(Duration.seconds(1), e -> {
            if (!countDownCheck.isFinished()) {
                countDownCheck.tick(1000);
                countdownLabel.setText(countDownCheck.getFormattedRemaining());
            } else {
                countdownTimeline.stop();
                countdownLabel.setText("00:00:00");
            }
        }));
        countdownTimeline.setCycleCount(Timeline.INDEFINITE);
        countdownTimeline.play();
    }

    /**
     * Resets the countdown timer to its initial state and restarts it.
     * Updates the provided label with the new formatted remaining time.
     *
     * @param countdownLabel the {@link Label} to display the formatted remaining time
     */
    public void resetCountdownTimer(Label countdownLabel) {
        System.out.println("Resetting countdown timer...");
        if (countdownTimeline != null) {
            countdownTimeline.stop();
        }
        countDownCheck.stop();
        countDownCheck.start();
        countdownLabel.setText(countDownCheck.getFormattedRemaining());
        startCountdownTimer(countdownLabel);
    }

    /**
     * Stops the countdown timer and halts the countdown check.
     */
    public void stopCountdownTimer() {
        if (countdownTimeline != null) {
            countdownTimeline.stop();
        }
        countDownCheck.stop();
    }

    /**
     * Handles power toggle events by turning the system power on or off.
     * Updates the button text to reflect the current power state.
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handlePowerToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.turnPowerOn();
            button.setText("Power: ON");
        } else {
            systemOperation.turnPowerOff();
            button.setText("Power: OFF");
        }
    }

    /**
     * Handles MPU sensor toggle events by enabling or disabling the MPU6050 sensor.
     * Updates the button text to reflect the current MPU state.
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handleMPUToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.enableMPU();
            button.setText("MPU: ON");
        } else {
            systemOperation.disableMPU();
            button.setText("MPU: OFF");
        }
    }

    /**
     * Handles telemetry toggle events by enabling or disabling telemetry.
     * Updates the button text to reflect the telemetry rate (TLM 500 or TLM 20).
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handleTLMToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.enableTLM();
            button.setText("TLM: 500");
        } else {
            systemOperation.disableTLM();
            button.setText("TLM: 20");
        }
    }

    /**
     * Handles GPS toggle events by enabling or disabling the GPS sensor.
     * Updates the button text to reflect the current GPS state.
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handleGPSToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.enableGPS();
            button.setText("GPS: ON");
        } else {
            systemOperation.disableGPS();
            button.setText("GPS: OFF");
        }
    }

    /**
     * Handles BMP sensor toggle events by enabling or disabling the BMP180 sensor.
     * Updates the button text to reflect the current BMP state.
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handleBMPToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.enableBMP();
            button.setText("BMP: ON");
        } else {
            systemOperation.disableBMP();
            button.setText("BMP: OFF");
        }
    }

    /**
     * Handles recording toggle events by enabling or disabling data recording.
     * Updates the button text to reflect the recording state ("Record" or "Stop").
     *
     * @param button the {@link ToggleButton} that triggered the event
     */
    public void handleRecordToggle(ToggleButton button) {
        if (button.isSelected()) {
            systemOperation.enableRecord();
            button.setText("Stop");
        } else {
            systemOperation.disableRecord();
            button.setText("Record");
        }
    }

    /**
     * Turns the system power on.
     */
    public void powerOn() {
        systemOperation.turnPowerOn();
        System.out.println("Power turned ON");
    }

    /**
     * Turns the system power off.
     */
    public void powerOff() {
        systemOperation.turnPowerOff();
        System.out.println("Power turned OFF");
    }

    /**
     * Checks the current power state of the system.
     *
     * @return true if the system power is on, false otherwise
     */
    public boolean isPowerOn() {
        return systemOperation.isPowerOn();
    }
}