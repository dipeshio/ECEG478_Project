/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 5/2/25
 * Time: 20:06
 *
 * Project: csci205_final_project
 * Package: AirForce.model
 * Class: CountDownCheckTest
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CountDownCheckTest {

    private CountDownCheck countdown;

    @BeforeEach
    public void setUp() {
        countdown = new CountDownCheck(60000);
    }

    @Test
    public void testFormattedRemaining() { // Make method public!
        countdown = new CountDownCheck(3661000);
        countdown.start();
        assertEquals("01:01:01", countdown.getFormattedRemaining());
    }
}

