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
 * Class: Dashboard.TimeLine
 *
 * Description:
 *
 * ****************************************
 */
package AirForce.model;
import java.util.ArrayList;
import java.util.List;

/**
 * The Timeline class manages a list of events for a rocket launch.
 * It uses an inner class, TimelineEvent, to represent individual events.
 */
public class TimeLine {

    /**
     * @param timeMarker e.g., "T-3h", "T+2h"
     */ // Inner class representing a single event.
        public record TimeLineEvent(String title, String description, String timeMarker) {
    }

    // List to store timeline events
    private final List<TimeLineEvent> events;

    /**
     * Constructor initializing the timeline's events list.
     */
    public TimeLine() {
        events = new ArrayList<>();
    }


    public void addEvent(String title, String description, String timeMarker) {
        TimeLineEvent event = new TimeLineEvent(title, description, timeMarker);
        events.add(event);
    }

    /**
     * Removes the first event that matches the given title (ignores case).
     */

    public void removeEvent(String title) {
        // Using a loop to find and remove the event by title.
        for (int i = 0; i < events.size(); i++) {
            if (events.get(i).title().equalsIgnoreCase(title)) {
                events.remove(i);
                return;
            }
        }
    }

    /**
     * Returns a list of all current timeline events.
     *
     *
     */
    public List<TimeLineEvent> getEvents() {
        return events;
    }

    /**
     * Displays the timeline events to the console.
     */
    public void displayTimeline() {
        for (TimeLineEvent event : events) {
            System.out.println(event.timeMarker() + " - " + event.title() + ": " + event.description()); // via chatGpt
        }
    }

    /**
     * Clears all events from the timeline.
     */
    public void clearTimeline() {
        events.clear();
    }

}

