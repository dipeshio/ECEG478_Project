/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper
 * Date: 4/14/25
 * Time: 2:05 PM
 *
 * Project: csci205_final_project
 * Package: PACKAGE_NAME
 * Class: Dashboard.SystemOperation
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.model;

/**
 * A class representing the operational state of the system in the AirForce application.
 * Manages the power state, voltage, and enablement of various system components such as
 * MPU6050 sensor, telemetry (TLM), GPS, BMP180 sensor, and data recording.
 */
public class SystemOperation {
    private boolean powerOn;      // Indicates if the system power is on
    private double voltage;       // Current system voltage
    private boolean mpuEnabled;   // Indicates if the MPU6050 sensor is enabled
    private boolean tlmEnabled;   // Indicates if telemetry is enabled
    private boolean gpsEnabled;   // Indicates if the GPS sensor is enabled
    private boolean bmpEnabled;   // Indicates if the BMP180 sensor is enabled
    private boolean record;       // Indicates if data recording is enabled

    /**
     * Constructs a SystemOperation instance with all components initially disabled
     * and voltage set to zero.
     */
    public SystemOperation() {
        this.powerOn = false;
        this.record = false;
        this.voltage = 0.0;
        this.mpuEnabled = false;
        this.tlmEnabled = false;
        this.gpsEnabled = false;
        this.bmpEnabled = false;
    }

    /**
     * Turns the system power on.
     */
    public void turnPowerOn() { this.powerOn = true; }

    /**
     * Turns the system power off.
     */
    public void turnPowerOff() { this.powerOn = false; }

    /**
     * Checks if the system power is on.
     *
     * @return true if the system power is on, false otherwise
     */
    public boolean isPowerOn() { return powerOn; }

    /**
     * Sets the system voltage.
     *
     * @param volts the voltage value to set
     */
    public void setVoltage(double volts) { this.voltage = volts; }

    /**
     * Returns the current system voltage.
     *
     * @return the current voltage
     */
    public double getVoltage() { return voltage; }

    /**
     * Enables the MPU6050 sensor.
     */
    public void enableMPU() { this.mpuEnabled = true; }

    /**
     * Disables the MPU6050 sensor.
     */
    public void disableMPU() { this.mpuEnabled = false; }

    /**
     * Checks if the MPU6050 sensor is enabled.
     *
     * @return true if the MPU6050 sensor is enabled, false otherwise
     */
    public boolean isMPUEnabled() { return mpuEnabled; }

    /**
     * Enables telemetry (TLM).
     */
    public void enableTLM() { this.tlmEnabled = true; }

    /**
     * Disables telemetry (TLM).
     */
    public void disableTLM() { this.tlmEnabled = false; }

    /**
     * Checks if telemetry (TLM) is enabled.
     *
     * @return true if telemetry is enabled, false otherwise
     */
    public boolean isTLMEnabled() { return tlmEnabled; }

    /**
     * Enables the GPS sensor.
     */
    public void enableGPS() { this.gpsEnabled = true; }

    /**
     * Disables the GPS sensor.
     */
    public void disableGPS() { this.gpsEnabled = false; }

    /**
     * Checks if the GPS sensor is enabled.
     *
     * @return true if the GPS sensor is enabled, false otherwise
     */
    public boolean isGPSEnabled() { return gpsEnabled; }

    /**
     * Enables the BMP180 sensor.
     */
    public void enableBMP() { this.bmpEnabled = true; }

    /**
     * Disables the BMP180 sensor.
     */
    public void disableBMP() { this.bmpEnabled = false; }

    /**
     * Checks if the BMP180 sensor is enabled.
     *
     * @return true if the BMP180 sensor is enabled, false otherwise
     */
    public boolean isBMPEnabled() { return bmpEnabled; }

    /**
     * Enables data recording.
     */
    public void enableRecord() { this.record = true;}

    /**
     * Disables data recording.
     */
    public void disableRecord() { this.record = false; }

    /**
     * Checks if data recording is enabled.
     *
     * @return true if data recording is enabled, false otherwise
     */
    public boolean recordON() { return record; }
}

