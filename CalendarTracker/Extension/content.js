(() => {
    /**
     * Content Script for Shift Scheduler Optimizer
     * Scrapes shift data from Bucknell Conportal
     */

    // Helper to parse date from header
    function parseCurrentDate() {
        // Look for the date in the header or specific element
        // Based on typical Conportal layout
        const dateElement = document.querySelector('.date-header, h2, h3');
        // Fallback: try to find a date string in the page

        // For now, let's assume the standard Conportal format if we can find it
        // Or use the calendar header we found in navigation

        // Better approach: The shift table usually has a date or the page URL might have it
        // URL format: show_shifts.php?date=2026-01-20
        const urlParams = new URLSearchParams(window.location.search);
        const dateParam = urlParams.get('date');

        if (dateParam) {
            const [year, month, day] = dateParam.split('-').map(Number);
            return {
                date: new Date(year, month - 1, day),
                dateString: dateParam
            };
        }

        // Fallback to scraping the header if URL param is missing
        // This part might need adjustment based on actual DOM
        const header = document.querySelector('td.main_title');
        if (header) {
            const text = header.textContent.trim();
            // Try to parse "Monday, January 20, 2026"
            const date = new Date(text);
            if (!isNaN(date.getTime())) {
                return {
                    date: date,
                    dateString: date.toISOString().split('T')[0]
                };
            }
        }

        // If all else fails, default to today (risky but better than crash)
        const today = new Date();
        return {
            date: today,
            dateString: today.toISOString().split('T')[0]
        };
    }

    // Helper to parse time string (e.g. "10:00 AM")
    function parseTime(timeStr) {
        if (!timeStr) return null;
        const [time, period] = timeStr.split(' ');
        let [hours, minutes] = time.split(':').map(Number);

        if (period === 'PM' && hours !== 12) hours += 12;
        if (period === 'AM' && hours === 12) hours = 0;

        return {
            hours,
            minutes,
            totalMinutes: hours * 60 + minutes,
            display: timeStr
        };
    }

    // Helper to determine shift status
    function getShiftStatus(element) {
        const text = element.textContent.toLowerCase();
        const className = element.className.toLowerCase();

        if (className.includes('taken') || text.includes('taken')) return 'taken';
        if (className.includes('dropped') || text.includes('dropped')) return 'dropped';
        if (className.includes('yours') || text.includes('your shift')) return 'your_shift';
        return 'available';
    }

    // Parse individual shift element
    function parseShiftContent(element) {
        // This depends heavily on the specific HTML structure of Conportal
        // Assuming shifts are in <div> or <td> elements with specific classes

        const text = element.textContent.trim();
        // Regex to extract time: "10:00 AM - 12:00 PM"
        const timeMatch = text.match(/(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)/i);

        if (!timeMatch) return null;

        const startTime = parseTime(timeMatch[1]);
        const endTime = parseTime(timeMatch[2]);

        // Extract role/location if available
        const roleMatch = text.match(/(Consultant|Leader|Tech)/i);
        const role = roleMatch ? roleMatch[1] : 'Consultant';

        // Extract assignee if taken
        // Look for text that isn't time or role
        let assignee = null;
        if (getShiftStatus(element) === 'taken') {
            const parts = text.split('\n');
            // Heuristic: assignee is usually the last line or distinct from time
            assignee = parts[parts.length - 1].trim();
        }

        let durationMinutes = endTime.totalMinutes - startTime.totalMinutes;
        if (durationMinutes < 0) durationMinutes += 24 * 60; // Handle overnight shifts
        const durationHours = durationMinutes / 60;

        return {
            startTime: startTime.display,
            endTime: endTime.display,
            startMinutes: startTime.totalMinutes,
            endMinutes: endTime.totalMinutes,
            durationHours,
            role,
            assignee
        };
    }

    /**
     * Scrape all shifts from the current page
     */
    function scrapeShifts() {
        const dateInfo = parseCurrentDate();
        const shifts = [];

        // Select all shift containers
        // Adjust selector based on actual DOM
        const shiftElements = document.querySelectorAll('.shift_cell, .shift_box, td[bgcolor]');

        shiftElements.forEach(el => {
            const content = parseShiftContent(el);
            if (content) {
                const status = getShiftStatus(el);

                shifts.push({
                    id: `${dateInfo.dateString}_${content.startTime}_${content.role}_${content.assignee || 'open'}`, // Unique ID
                    date: dateInfo.dateString,
                    ...content,
                    status
                });
            }
        });

        return {
            date: dateInfo.dateString,
            currentDate: dateInfo,
            shifts
        };
    }

    /**
     * Save scraped data to chrome.storage
     */
    async function saveShiftData(data) {
        return new Promise((resolve) => {
            chrome.storage.local.get(['allShiftData'], (result) => {
                const allData = result.allShiftData || {};

                // Update data for this date
                allData[data.date] = data.shifts;

                chrome.storage.local.set({ allShiftData: allData }, () => {
                    console.log(`Saved ${data.shifts.length} shifts for ${data.date}`);
                    resolve(true);
                });
            });
        });
    }

    /**
     * Check if date is MLK Holiday (Jan 19, 2026)
     */
    function isMLKHoliday(dateInfo) {
        if (!dateInfo || !dateInfo.date) return false;

        const date = dateInfo.date;
        // Check for January 19th, 2026 (Monday, MLK Day)
        return date.getMonth() === 0 && date.getDate() === 19 && date.getFullYear() === 2026;
    }

    /**
     * Main scraping function
     */
    async function main() {
        console.log('Shift Scheduler Optimizer: Starting scrape...');

        const data = scrapeShifts();

        if (!data) {
            console.error('Failed to scrape shift data');
            return;
        }

        // Check for MLK Holiday edge case
        if (isMLKHoliday(data.currentDate)) {
            console.warn('MLK Holiday (January 19th) detected. Shifts may not be available for selection.');
            data.isHoliday = true;
            data.holidayMessage = 'January 19th is MLK Day. Please visit January 26th for accurate future scheduling.';
        }

        // Save the data
        const saved = await saveShiftData(data);

        if (saved) {
            // Notify background script
            chrome.runtime.sendMessage({
                action: 'SHIFTS_SCRAPED',
                data: {
                    date: data.currentDate.dateString,
                    shiftCount: data.shifts.length,
                    isHoliday: data.isHoliday || false
                }
            });
        }

        console.log('Shift Scheduler Optimizer: Scrape complete');
    }

    // Run when page is loaded
    if (document.readyState === 'complete') {
        main();
    } else {
        window.addEventListener('load', main);
    }

    // Listen for messages from popup or background
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.action === 'SCRAPE_NOW') {
            main().then(() => {
                sendResponse({ success: true });
            });
            return true; // Keep message channel open for async response
        }

        if (message.action === 'NAVIGATE_TO_DATE') {
            navigateToDate(message.date).then((success) => {
                if (success) {
                    // Wait for page update then scrape
                    setTimeout(() => {
                        main();
                    }, 1500); // 1.5s delay for page load
                } else {
                    console.error('Navigation failed, aborting scrape for this date.');
                    // Optionally notify background of failure so it doesn't hang?
                    // But background loop relies on SHIFTS_SCRAPED. 
                    // If we fail, we should probably send a failure message or just skip.
                    // For now, logging error is enough to stop the infinite loop of "success".
                }
            });
            return true;
        }

        if (message.action === 'GET_PAGE_DATA') {
            const data = scrapeShifts();
            sendResponse(data);
            return true;
        }
    });

    /**
     * Navigate the calendar to a specific date
     */
    async function navigateToDate(targetDateStr) {
        const targetDate = new Date(targetDateStr + 'T12:00:00');
        const targetMonth = targetDate.getMonth();
        const targetYear = targetDate.getFullYear();
        const targetDay = targetDate.getDate();

        console.log(`Navigating to ${targetDateStr}...`);

        // 1. Find Calendar Header (Month Year)
        // Based on screenshot: "December 2025" inside a container with arrows
        // We look for the text that matches a month name
        const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

        // Helper to parse header text
        const getCalendarState = () => {
            const allDivs = document.querySelectorAll('div, span, td, th'); // Broad search
            for (const el of allDivs) {
                const text = el.textContent.trim();
                // Match "Month Year" e.g., "December 2025"
                const match = text.match(new RegExp(`^(${months.join('|')})\\s+(\\d{4})$`));
                if (match) {
                    return {
                        element: el,
                        monthIndex: months.indexOf(match[1]),
                        year: parseInt(match[2])
                    };
                }
            }
            return null;
        };

        let calendarState = getCalendarState();
        if (!calendarState) {
            console.error('Could not find calendar header');
            return false;
        }

        // 2. Navigate Month/Year
        let maxAttempts = 24; // Prevent infinite loops
        while (maxAttempts > 0) {
            if (calendarState.year === targetYear && calendarState.monthIndex === targetMonth) {
                break; // We are in the right month
            }

            // Determine direction
            const isTargetFuture = (targetYear > calendarState.year) ||
                (targetYear === calendarState.year && targetMonth > calendarState.monthIndex);

            // Find arrows relative to the header
            // Based on screenshot: arrows are siblings or close neighbors
            // We look for elements with text "←", "→", "<", ">" or specific classes if known
            // Heuristic: Look for clickable elements near the header
            const container = calendarState.element.parentElement;
            const arrows = container.querySelectorAll('a, span, div'); // Potential arrow elements

            let clicked = false;
            for (const arrow of arrows) {
                const text = arrow.textContent.trim();
                if (isTargetFuture && (text.includes('→') || text.includes('>'))) {
                    arrow.click();
                    clicked = true;
                    break;
                } else if (!isTargetFuture && (text.includes('←') || text.includes('<'))) {
                    arrow.click();
                    clicked = true;
                    break;
                }
            }

            if (!clicked) {
                console.error('Could not find navigation arrows');
                return false;
            }

            // Wait for update
            await new Promise(r => setTimeout(r, 500));
            calendarState = getCalendarState(); // Re-check
            maxAttempts--;
        }

        // 3. Click the Day
        // Find the day number in the calendar grid
        // We need to be careful not to click prev/next month days (often grayed out)
        // Heuristic: Find cell with exact number text, check if it looks "active"

        // Re-find container as DOM might have changed
        calendarState = getCalendarState();
        if (!calendarState) return false;

        // The calendar grid should be near the header
        // We search in the vicinity
        let calendarContainer = calendarState.element.parentElement.parentElement; // Go up a bit

        // Find all elements with the day number
        const dayElements = Array.from(calendarContainer.querySelectorAll('td, div, span, a'))
            .filter(el => {
                const text = el.textContent.trim();
                return text === String(targetDay) &&
                    // Ensure it's a day number (short text)
                    text.length <= 2 &&
                    // Ensure it's visible
                    el.offsetParent !== null;
            });

        // Filter out "gray" days (prev/next month)
        // Usually they have a class like 'other-month', 'gray', 'inactive' or inline style opacity
        // If multiple found, we pick the one that looks "main"
        // For now, we assume the "middle" one if multiple, or the one without specific "other" classes

        for (const el of dayElements) {
            // Check for common "inactive" class names
            const cls = el.className.toLowerCase();
            if (cls.includes('prev') || cls.includes('next') || cls.includes('other') || cls.includes('gray')) {
                continue;
            }

            // Click it!
            console.log(`Clicking day ${targetDay}`);
            el.click();
            return true;
        }

        console.error(`Could not find day ${targetDay} to click`);
        return false;
    }
})();
