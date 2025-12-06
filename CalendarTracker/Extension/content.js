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
            if (!success) {
                console.error('Navigation failed or waiting for reload.');
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
 * Returns true if an action was taken (click), false if failed
 */
async function navigateToDate(targetDateStr) {
    // Parse target date (handle timezone by treating string as local date)
    const [tYear, tMonth, tDay] = targetDateStr.split('-').map(Number);
    // Note: tMonth is 1-based (01-12), JS Date month is 0-based (0-11)

    console.log(`Navigating to ${targetDateStr}...`);

    // 1. Find Calendar Header (Month Year)
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

    // Helper to parse header text
    const getCalendarState = () => {
        const allDivs = document.querySelectorAll('td.main_title, div, span, h2, h3'); // Broad search
        for (const el of allDivs) {
            const text = el.textContent.trim();
            // Match "Month Year" e.g., "December 2025"
            const match = text.match(new RegExp(`^(${months.join('|')})\\s+(\\d{4})$`));
            if (match) {
                return {
                    element: el,
                    monthIndex: months.indexOf(match[1]), // 0-11
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

    console.log(`Current Calendar: ${months[calendarState.monthIndex]} ${calendarState.year}`);

    // 2. Navigate Month/Year
    // Check if we are in the right month
    if (calendarState.year !== tYear || calendarState.monthIndex !== (tMonth - 1)) {
        // Determine direction
        const isTargetFuture = (tYear > calendarState.year) ||
            (tYear === calendarState.year && (tMonth - 1) > calendarState.monthIndex);

        console.log(`Target is ${isTargetFuture ? 'Future' : 'Past'}. Finding arrow...`);

        // Find arrows relative to the header
        const container = calendarState.element.parentElement;
        // Look for links or clickable elements that might be arrows
        const arrows = container.querySelectorAll('a, span, div, img');

        for (const arrow of arrows) {
            const text = arrow.textContent.trim();
            const alt = arrow.getAttribute('alt') || '';
            const title = arrow.getAttribute('title') || '';

            // Check for typical arrow symbols or text
            const isNext = text.includes('→') || text.includes('>') || text.includes('Next') || alt.includes('Next') || title.includes('Next');
            const isPrev = text.includes('←') || text.includes('<') || text.includes('Prev') || alt.includes('Prev') || title.includes('Prev');

            if (isTargetFuture && isNext) {
                console.log('Clicking Next Month');
                arrow.click();
                return true; // Action taken
            } else if (!isTargetFuture && isPrev) {
                console.log('Clicking Prev Month');
                arrow.click();
                return true; // Action taken
            }
        }

        console.error('Could not find appropriate navigation arrow');
        return false;
    }

    // 3. Click the Day
    // We are in the correct month, now find the day
    console.log(`In correct month. Finding day ${tDay}...`);

    // Re-find container as DOM might have changed (unlikely if we didn't reload, but safe)
    calendarState = getCalendarState();
    if (!calendarState) return false;

    // The calendar grid should be near the header
    // Go up a few levels to find the table/grid
    let calendarContainer = calendarState.element.closest('table') || calendarState.element.parentElement.parentElement;

    // Find all elements with the day number
    const dayElements = Array.from(calendarContainer.querySelectorAll('td, div, span, a'))
        .filter(el => {
            const text = el.textContent.trim();
            return text === String(tDay) &&
                // Ensure it's a day number (short text)
                text.length <= 2 &&
                // Ensure it's visible
                el.offsetParent !== null;
        });

    // Filter out "gray" days (prev/next month)
    for (const el of dayElements) {
        // Check for common "inactive" class names or styles
        const cls = el.className.toLowerCase();
        const style = el.getAttribute('style') || '';

        if (cls.includes('prev') || cls.includes('next') || cls.includes('other') || cls.includes('gray') || style.includes('opacity')) {
            continue;
        }

        // Check if it's a link or has a link child (days are usually links)
        const link = el.tagName === 'A' ? el : el.querySelector('a');

        if (link) {
            console.log(`Clicking day ${tDay} (link)`);
            link.click();
            return true;
        } else {
            // Try clicking the element itself
            console.log(`Clicking day ${tDay} (cell)`);
            el.click();
            return true;
        }
    }

    console.error(`Could not find day ${tDay} to click`);
    return false;
}
}) ();
