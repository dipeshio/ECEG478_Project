/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper
 * Date: 4/14/25
 * Time: 1:30 PM
 *
 * Project: csci205_final_project
 * Class: TransformData
 *
 * Description:
 *
 * ****************************************
 */


package AirForce.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The Location class tracks the rocket’s position (latitude/longitude),
 * its velocity, altitude, and keeps a history of past positions.
 */
public class Location {
    private double latitude;
    private double longitude;
    private double velocity;
    private double altitude;
    private final List<Coordinate> trajectory;

    // Initialize with coordinates from WeatherAPI
    public Location(String defaultLocation) {
        this.trajectory = new ArrayList<>();
        // Fetch initial coordinates from WeatherAPI
        try {
            WeatherAPI weatherAPI = new WeatherAPI();
            Coordinates coords = fetchCoordinates(weatherAPI, defaultLocation);
            this.latitude = coords.latitude;
            this.longitude = coords.longitude;
        } catch (IOException | InterruptedException e) {
            System.err.println("Error fetching weather data: " + e.getMessage());
            this.latitude = 0.0;
            this.longitude = 0.0;
        }
        this.velocity = 0.0;
        this.altitude = 0.0;
        this.trajectory.add(new Coordinate(latitude, longitude));
    }

    // Helper method to fetch coordinates from WeatherAPI
    private Coordinates fetchCoordinates(WeatherAPI weatherAPI, String location) throws IOException, InterruptedException {
        String weatherData = weatherAPI.fetchWeatherData(location);
        if (weatherData.startsWith("Weather: Location not found") ||
                weatherData.startsWith("Weather: API Error") ||
                weatherData.startsWith("Weather: Error")) {
            return new Coordinates(0.0, 0.0);
        }

        // Call extractCoordinates and convert WeatherAPI.Coordinates to Location.Coordinates
        WeatherAPI.Coordinates weatherCoords = weatherAPI.extractCoordinates(location);
        return new Coordinates(weatherCoords.latitude, weatherCoords.longitude);
    }

    // Update latitude and longitude, and add the new point to the trajectory
    public void setLocation(double latitude, double longitude) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.trajectory.add(new Coordinate(latitude, longitude));
    }

    // Update the rocket’s current velocity
    public void setVelocity(double velocity) {
        this.velocity = velocity;
    }

    // Update the rocket’s current altitude
    public void setAltitude(double altitude) {
        this.altitude = altitude;
    }

    // Get the current latitude
    public double getLatitude() {
        return latitude;
    }

    // Get the current longitude
    public double getLongitude() {
        return longitude;
    }

    // Get the current velocity
    public double getVelocity() {
        return velocity;
    }

    // Get the current altitude
    public double getAltitude() {
        return altitude;
    }

    // Return a copy of the trajectory history
    public List<Coordinate> getTrajectory() {
        return new ArrayList<>(trajectory);
    }

    /**
         * Simple data class for a latitude/longitude pair.
         */
        public record Coordinate(double latitude, double longitude) {
        // Store the given latitude and longitude
    }

    /**
         * Helper class to hold latitude and longitude.
         */
        private record Coordinates(double latitude, double longitude) {
    }
}
