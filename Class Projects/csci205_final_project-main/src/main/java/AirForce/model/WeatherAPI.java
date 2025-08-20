/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/28/25
 * Time: 01:17
 *
 * Project: csci205_final_project
 * Package: AirForce.model
 * Class: WeatherAPI
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.model;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * A utility class for fetching weather data from the OpenWeatherMap API in the AirForce application.
 * Provides methods to retrieve weather information and coordinates for a specified location.
 * Uses HTTP requests to interact with the API and parses JSON responses manually.
 */
public class WeatherAPI {
    private static final String API_KEY = "cbd80e60ba4d19ea62626c83fd2eb8e0"; // OpenWeatherMap API key


    /**
     * Fetches weather data for the specified location using the OpenWeatherMap API.
     * Retrieves coordinates via the Geocoding API, then fetches weather details including
     * temperature, humidity, visibility, wind speed, and sunset time.
     *
     * @param location the location for which to fetch weather data (e.g., "Lewisburg,PA,US")
     * @return a formatted string containing weather data, or an error message if the request fails
     * @throws IOException if an I/O error occurs during the HTTP request
     * @throws InterruptedException if the HTTP request is interrupted
     */
    public String fetchWeatherData(String location) throws IOException, InterruptedException {
        // Step 1: Fetch coordinates using Geocoding API
        String geocodingUrl = String.format("https://api.openweathermap.org/geo/1.0/direct?q=%s&limit=1&appid=%s", location, API_KEY);

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest geocodingRequest = HttpRequest.newBuilder()
                .uri(URI.create(geocodingUrl))
                .build();
        HttpResponse<String> geocodingResponse = client.send(geocodingRequest, HttpResponse.BodyHandlers.ofString());
        String geocodingBody = geocodingResponse.body();

        // Log the raw response for debugging
        System.out.println("Geocoding API Response: " + geocodingBody);

        // Parse Geocoding JSON manually
        if (geocodingBody.equals("[]")) {
            return "Weather: Location not found";
        }

        // Check for error responses
        if (geocodingBody.contains("\"cod\":") && geocodingBody.contains("\"message\":")) {
            return "Weather: API Error - Check API Key or Rate Limits";
        }

        double lat, lon;
        try {
            // Extract latitude
            String latStr = getString(geocodingBody);
            System.out.println("Parsed latStr: '" + latStr + "'");

            // Extract longitude
            String lonStr = getLonStr(geocodingBody);
            System.out.println("Parsed lonStr: '" + lonStr + "'");

            // Parse the values
            lat = Double.parseDouble(latStr);
            lon = Double.parseDouble(lonStr);
        } catch (Exception e) {
            System.out.println("Parsing error: " + e.getMessage());
            return "Weather: Error parsing coordinates";
        }

        // Step 2: Fetch weather data using the coordinates
        String weatherUrl = String.format("https://api.openweathermap.org/data/2.5/weather?lat=%s&lon=%s&appid=%s&units=metric", lat, lon, API_KEY);
        HttpRequest weatherRequest = HttpRequest.newBuilder()
                .uri(URI.create(weatherUrl))
                .build();
        HttpResponse<String> weatherResponse = client.send(weatherRequest, HttpResponse.BodyHandlers.ofString());
        String weatherBody = weatherResponse.body();

        // Log the weather API response for debugging
        System.out.println("Weather API Response: " + weatherBody);

        // Parse Weather JSON manually
        double temp, humidity, visibility, windSpeed;
        String sunriseStr = "N/A", sunsetStr;
        try {
            // Check for error responses
            if (weatherBody.contains("\"cod\":") && weatherBody.contains("\"message\":")) {
                int messageStart = weatherBody.indexOf("\"message\":\"") + 11;
                int messageEnd = weatherBody.indexOf("\"", messageStart);
                String errorMessage = weatherBody.substring(messageStart, messageEnd);
                return "Weather: API Error - " + errorMessage;
            }

            // Extract temperature
            int tempStart = weatherBody.indexOf("\"temp\":") + 7;
            int tempEnd = weatherBody.indexOf(",", tempStart);
            if (tempStart < 7 || tempEnd == -1 || tempStart >= tempEnd) {
                System.out.println("Invalid temperature format: tempStart=" + tempStart + ", tempEnd=" + tempEnd + ". Using default 0.0.");
                temp = 0.0;
            } else {
                String tempStr = weatherBody.substring(tempStart, tempEnd).trim();
                System.out.println("Parsed tempStr: '" + tempStr + "'");
                try {
                    temp = Double.parseDouble(tempStr);
                } catch (NumberFormatException e) {
                    System.out.println("Failed to parse temperature: " + e.getMessage() + ". Using default 0.0.");
                    temp = 0.0;
                }
            }

            // Extract humidity
            int humidityStart = weatherBody.indexOf("\"humidity\":") + 11;
            int humidityEnd = weatherBody.indexOf(",", humidityStart);
            if (humidityStart < 11 || humidityEnd == -1 || humidityStart >= humidityEnd) {
                System.out.println("Invalid humidity format: humidityStart=" + humidityStart + ", humidityEnd=" + humidityEnd + ". Using default 0.0.");
                humidity = 0.0;
            } else {
                String humidityStr = weatherBody.substring(humidityStart, humidityEnd).trim();
                System.out.println("Parsed humidityStr: '" + humidityStr + "'");
                try {
                    humidity = Double.parseDouble(humidityStr);
                } catch (NumberFormatException e) {
                    System.out.println("Failed to parse humidity: " + e.getMessage() + ". Using default 0.0.");
                    humidity = 0.0;
                }
            }

            // Extract visibility
            int visibilityStart = weatherBody.indexOf("\"visibility\":") + 12;
            int visibilityEnd = weatherBody.indexOf(",", visibilityStart);
            if (visibilityEnd == -1) visibilityEnd = weatherBody.indexOf("}", visibilityStart);
            if (visibilityStart < 12 || visibilityEnd == -1 || visibilityStart >= visibilityEnd) {
                System.out.println("Invalid visibility format: visibilityStart=" + visibilityStart + ", visibilityEnd=" + visibilityEnd + ". Using default 0.0.");
                visibility = 0.0;
            } else {
                String visibilityStr = weatherBody.substring(visibilityStart, visibilityEnd).trim();
                System.out.println("Parsed visibilityStr: '" + visibilityStr + "'");
                try {
                    visibility = Double.parseDouble(visibilityStr);
                } catch (NumberFormatException e) {
                    System.out.println("Failed to parse visibility: " + e.getMessage() + ". Using default 0.0.");
                    visibility = 0.0;
                }
            }

            // Extract wind speed
            int windSpeedStart = weatherBody.indexOf("\"speed\":") + 8;
            int windSpeedEnd = weatherBody.indexOf(",", windSpeedStart);
            if (windSpeedEnd == -1) windSpeedEnd = weatherBody.indexOf("}", windSpeedStart);
            if (windSpeedStart < 8 || windSpeedEnd == -1 || windSpeedStart >= windSpeedEnd) {
                System.out.println("Invalid wind speed format: windSpeedStart=" + windSpeedStart + ", windSpeedEnd=" + windSpeedEnd + ". Using default 0.0.");
                windSpeed = 0.0;
            } else {
                String windSpeedStr = weatherBody.substring(windSpeedStart, windSpeedEnd).trim();
                System.out.println("Parsed windSpeedStr: '" + windSpeedStr + "'");
                try {
                    windSpeed = Double.parseDouble(windSpeedStr);
                } catch (NumberFormatException e) {
                    System.out.println("Failed to parse wind speed: " + e.getMessage() + ". Using default 0.0.");
                    windSpeed = 0.0;
                }
            }

            // Extract timezone
            long timezoneOffset;
            int timezoneStart = weatherBody.indexOf("\"timezone\":") + 11;
            int timezoneEnd = weatherBody.indexOf(",", timezoneStart);
            if (timezoneEnd == -1) timezoneEnd = weatherBody.indexOf("}", timezoneStart);
            if (timezoneStart < 11 || timezoneEnd == -1 || timezoneStart >= timezoneEnd) {
                System.out.println("Invalid timezone format: timezoneStart=" + timezoneStart + ", timezoneEnd=" + timezoneEnd + ". Using default 0.");
                timezoneOffset = 0;
            } else {
                String timezoneStr = weatherBody.substring(timezoneStart, timezoneEnd).trim();
                System.out.println("Parsed timezoneStr: '" + timezoneStr + "'");
                try {
                    timezoneOffset = Long.parseLong(timezoneStr);
                } catch (NumberFormatException e) {
                    System.out.println("Failed to parse timezone: " + e.getMessage() + ". Using default 0.");
                    timezoneOffset = 0;
                }
            }

            // Extract sunset
            long sunsetTimestamp;
            int sunsetStart = weatherBody.indexOf("\"sunset\":") + 9;
            if (sunsetStart < 9) {
                System.out.println("Sunset field not found. Using default N/A.");
                sunsetStr = "N/A";
            } else {
                // Find the end of the numeric value
                int sunsetEnd = sunsetStart;
                while (sunsetEnd < weatherBody.length() && Character.isDigit(weatherBody.charAt(sunsetEnd))) {
                    sunsetEnd++;
                }
                System.out.println("sunsetStart: " + sunsetStart + ", sunsetEnd: " + sunsetEnd);
                if (sunsetStart >= sunsetEnd) {
                    System.out.println("Invalid sunset format: sunsetStart=" + sunsetStart + ", sunsetEnd=" + sunsetEnd + ". Using default N/A.");
                    sunsetStr = "N/A";
                } else {
                    String sunsetTimestampStr = weatherBody.substring(sunsetStart, sunsetEnd).trim();
                    System.out.println("Parsed sunsetTimestampStr: '" + sunsetTimestampStr + "'");
                    if (!sunsetTimestampStr.matches("\\d+")) {
                        System.out.println("Sunset timestamp contains invalid characters: '" + sunsetTimestampStr + "'. Using default N/A.");
                        sunsetStr = "N/A";
                    } else {
                        try {
                            sunsetTimestamp = Long.parseLong(sunsetTimestampStr);
                            java.time.Instant sunsetInstant = java.time.Instant.ofEpochSecond(sunsetTimestamp + timezoneOffset);
                            java.time.ZonedDateTime sunsetZdt = sunsetInstant.atZone(java.time.ZoneId.of("UTC"));
                            sunsetStr = sunsetZdt.format(java.time.format.DateTimeFormatter.ofPattern("hh:mm a"));
                            System.out.println("Formatted sunset: " + sunsetStr);
                        } catch (NumberFormatException e) {
                            System.out.println("Failed to parse sunset: " + e.getMessage() + ". Using default N/A.");
                            sunsetStr = "N/A";
                        }
                    }
                }
            }

            // Format the weather data
            String displayLocation = "Lewisburg PA";
            return String.format(
                    """
                            Location: %s
                            Temperature: %.2f°C, Humidity %.2f%%
                            Lat: %.4f, Lon: %.4f
                            Visibility: %.0f m
                            Wind Speed: %.1f m/s
                            Sunset: %s""",
                    displayLocation, temp, humidity,
                    lat, lon,
                    visibility,
                    windSpeed,
                    sunsetStr
            );
        } catch (Exception e) {
            System.out.println("Weather Parsing error: " + e.getMessage());
            e.printStackTrace();
            return "Weather: Error parsing weather data";
        }
    }

    private static String getLonStr(String geocodingBody) {
        String lonMarker = "\"lon\":";
        int lonStart = geocodingBody.indexOf(lonMarker) + lonMarker.length();
        int lonEnd = geocodingBody.indexOf(",", lonStart);
        if (lonEnd == -1) lonEnd = geocodingBody.indexOf("}", lonStart);
        if (lonStart < 0 || lonEnd < 0 || lonStart >= lonEnd) {
            throw new IllegalStateException("Could not find longitude in response");
        }
        return geocodingBody.substring(lonStart, lonEnd).trim();
    }

    private static String getString(String geocodingBody) {
        String latMarker = "\"lat\":";
        int latStart = geocodingBody.indexOf(latMarker) + latMarker.length();
        int latEnd = geocodingBody.indexOf(",", latStart);
        if (latEnd == -1) latEnd = geocodingBody.indexOf("}", latStart);
        if (latStart < 0 || latEnd < 0 || latStart >= latEnd) {
            throw new IllegalStateException("Could not find latitude in response");
        }
        return geocodingBody.substring(latStart, latEnd).trim();
    }

    /**
     * Extracts geographic coordinates (latitude and longitude) for the specified location
     * using the OpenWeatherMap Geocoding API.
     *
     * @param location the location for which to fetch coordinates (e.g., "Lewisburg,PA,US")
     * @return a {@link Coordinates} object containing the latitude and longitude, or (0.0, 0.0) if the request fails
     * @throws IOException if an I/O error occurs during the HTTP request
     * @throws InterruptedException if the HTTP request is interrupted
     */
    public Coordinates extractCoordinates(String location) throws IOException, InterruptedException {
        String geocodingUrl = String.format("https://api.openweathermap.org/geo/1.0/direct?q=%s&limit=1&appid=%s", location, API_KEY);

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest geocodingRequest = HttpRequest.newBuilder()
                .uri(URI.create(geocodingUrl))
                .build();
        HttpResponse<String> geocodingResponse = client.send(geocodingRequest, HttpResponse.BodyHandlers.ofString());
        String geocodingBody = geocodingResponse.body();

        if (geocodingBody.equals("[]")) {
            return new Coordinates(0.0, 0.0);
        }

        if (geocodingBody.contains("\"cod\":") && geocodingBody.contains("\"message\":")) {
            return new Coordinates(0.0, 0.0);
        }

        double lat, lon;
        try {
            // Extract latitude
            String latStr = getLatStr(geocodingBody);

            // Extract longitude
            String lonStr = getStr(geocodingBody);

            // Parse the values
            lat = Double.parseDouble(latStr);
            lon = Double.parseDouble(lonStr);
        } catch (Exception e) {
            System.out.println("Parsing error: " + e.getMessage());
            return new Coordinates(0.0, 0.0);
        }

        return new Coordinates(lat, lon);
    }


    private static String getStr(String geocodingBody) {
        String lonMarker = "\"lon\":";
        int lonStart = geocodingBody.indexOf(lonMarker) + lonMarker.length();
        int lonEnd = geocodingBody.indexOf(",", lonStart);
        if (lonEnd == -1) lonEnd = geocodingBody.indexOf("}", lonStart);
        if (lonStart < 0 || lonEnd < 0 || lonStart >= lonEnd) {
            throw new IllegalStateException("Could not find longitude in response");
        }
        return geocodingBody.substring(lonStart, lonEnd).trim();
    }

    private static String getLatStr(String geocodingBody) {
        String latMarker = "\"lat\":";
        int latStart = geocodingBody.indexOf(latMarker) + latMarker.length();
        int latEnd = geocodingBody.indexOf(",", latStart);
        if (latEnd == -1) latEnd = geocodingBody.indexOf("}", latStart);
        if (latStart < 0 || latEnd < 0 || latStart >= latEnd) {
            throw new IllegalStateException("Could not find latitude in response");
        }
        return geocodingBody.substring(latStart, latEnd).trim();
    }

    /**
     * A helper class to encapsulate geographic coordinates.
     */
    public static class Coordinates {
        public final double latitude;
        public final double longitude;

        Coordinates(double latitude, double longitude) {
            this.latitude = latitude;
            this.longitude = longitude;
        }
    }
}