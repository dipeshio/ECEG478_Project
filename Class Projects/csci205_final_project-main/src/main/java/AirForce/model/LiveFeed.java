/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/14/25
 * Time: 2:06 PM
 *
 * Project: csci205_final_project
 * Class: Dashboard.LiveFeed
 *
 * Description:
 *
 * ****************************************
 */
package AirForce.model;

import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;

/**
 * A utility class for managing a live video feed in the AirForce application.
 * Controls playback of a video file using JavaFX's {@link MediaPlayer} and displays
 * it in a {@link MediaView}. Supports playing, stopping, and shutting down the video feed.
 */
public class LiveFeed {
    private MediaPlayer mediaPlayer;   // Media player for controlling video playback
    private boolean isVideoPlaying;   // Indicates if the video is currently playing

    /**
     * Constructs a LiveFeed instance and initializes the video feed.
     * Loads a video file from the specified resource path and associates it with the provided {@link MediaView}.
     *
     * @param mediaPane the {@link MediaView} to display the video
     * @throws IllegalArgumentException if the video file is not found
     */
    public LiveFeed(MediaView mediaPane) {
        try {
            String resourcePath = "/LiveFeed.mp4";
            java.net.URL videoUrl = getClass().getResource(resourcePath);
            if (videoUrl == null) {
                throw new IllegalArgumentException("Video file not found at: " + resourcePath);
            }
            String videoPath = videoUrl.toExternalForm();
            Media media = new Media(videoPath);
            mediaPlayer = new MediaPlayer(media);
            mediaPane.setMediaPlayer(mediaPlayer);
            mediaPlayer.setCycleCount(MediaPlayer.INDEFINITE);
            mediaPlayer.setOnError(() -> System.out.println("Media error: " + mediaPlayer.getError().getMessage()));
        } catch (Exception e) {
            System.out.println("Failed to load video: " + e.getMessage());
        }
    }


    /**
     * Starts playback of the video feed if it is not already playing.
     */
    public void play() {
        if (mediaPlayer != null && !isVideoPlaying) {
            mediaPlayer.play();
            isVideoPlaying = true;
        }
    }

    /**
     * Stops playback of the video feed if it is currently playing.
     */
    public void stop() {
        if (mediaPlayer != null && isVideoPlaying) {
            mediaPlayer.stop();
            isVideoPlaying = false;
        }
    }

    /**
     * Shuts down the video feed by stopping playback and disposing of the media player.
     */
    public void shutdown() {
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            mediaPlayer.dispose();
        }
    }

    /**
     * Checks if the video feed is currently playing.
     *
     * @return true if the video is playing, false otherwise
     */
    public boolean isVideoPlaying() {
        return isVideoPlaying;
    }
}