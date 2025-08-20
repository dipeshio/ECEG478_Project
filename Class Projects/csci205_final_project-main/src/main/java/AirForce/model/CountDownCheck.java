
/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper and SooAh Lay
 * Date: 4/14/25
 * Time: 2:05 PM
 *
 * Project: csci205_final_project
 * Package: PACKAGE_NAME
 * Class: Dashboard.CountDownCheck
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.model;

/**
 * A utility class for managing a countdown timer in the AirForce application.
 * Tracks the remaining time of a specified duration, provides formatted time output,
 * and supports starting, stopping, and ticking the countdown.
 */
public class CountDownCheck {
    private final long durationMillis;    // Total duration of the countdown in milliseconds
    private long remainingMillis;         // Remaining time in milliseconds
    private boolean running;              // Indicates if the countdown is currently running

    /**
     * Constructs a CountDownCheck with the specified duration.
     *
     * @param durationMillis the total duration of the countdown in milliseconds
     */
    public CountDownCheck(long durationMillis) {
        this.durationMillis = durationMillis;
        this.remainingMillis = durationMillis;
        this.running = false;
    }

    /**
     * Starts the countdown by resetting the remaining time to the initial duration
     * and setting the running state to true.
     */
    public void start() {
        this.remainingMillis = durationMillis;
        this.running = true;
    }

    /**
     * Stops the countdown by setting the running state to false.
     */
    public void stop() {
        this.running = false;
    }

    /**
     * Decrements the remaining time by the specified delta, if the countdown is running.
     * Ensures the remaining time does not go below zero.
     *
     * @param deltaMillis the time to subtract in milliseconds
     */
    public void tick(long deltaMillis) {
        if (running) {
            remainingMillis = Math.max(0, remainingMillis - deltaMillis);
        }
    }

    /**
     * Returns the remaining time in milliseconds.
     * If the countdown is not running, returns zero.
     *
     * @return the remaining time in milliseconds
     */
    public long getRemainingMillis() {
        return running ? remainingMillis : 0;
    }

    /**
     * Returns the remaining time formatted as "HH:mm:ss".
     *
     * @return a string representing the remaining time in hours, minutes, and seconds
     */
    public String getFormattedRemaining() {
        long rem = getRemainingMillis() / 1000;
        long hours = rem / 3600;
        long minutes = (rem % 3600) / 60;
        long seconds = rem % 60;
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    /**
     * Checks if the countdown has finished (i.e., remaining time is zero).
     *
     * @return true if the countdown is finished, false otherwise
     */
    public boolean isFinished() {
        return getRemainingMillis() == 0;
    }
}