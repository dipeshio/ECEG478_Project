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

        // NAVIGATE_TO_DATE is no longer needed as background.js handles URL updates

        if (message.action === 'GET_PAGE_DATA') {
            const data = scrapeShifts();
            sendResponse(data);
            return true;
        }
    });
})();
