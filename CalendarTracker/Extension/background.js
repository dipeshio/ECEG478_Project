/**
 * Background Service Worker for Shift Scheduler Optimizer
 * Handles data persistence and state management
 */

// Bulk Scrape State
let bulkScrapeState = {
    isActive: false,
    queue: [],
    total: 0,
    current: 0,
    results: [],
    tabId: null,
    retryCount: 0
};

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.action) {
        case 'SHIFTS_SCRAPED':
            handleShiftsScraped(message.data);
            break;

        case 'START_BULK_SCRAPE':
            startBulkScrape(message.data);
            break;

        case 'GET_ALL_SHIFTS':
            getAllShifts().then(sendResponse);
            return true; // Keep channel open for async

        case 'CLEAR_DATA':
            clearAllData().then(sendResponse);
            return true;

        case 'GET_USER_CALENDAR':
            getUserCalendar().then(sendResponse);
            return true;

        case 'SAVE_USER_CALENDAR':
            saveUserCalendar(message.calendar).then(sendResponse);
            return true;

        case 'GET_STATS':
            getStats().then(sendResponse);
            return true;
    }
});

/**
 * Start the bulk scraping process
 */
function startBulkScrape({ startDate, endDate }) {
    console.log(`Starting bulk scrape from ${startDate} to ${endDate}`);

    // Generate date queue
    const queue = [];
    let current = new Date(startDate);
    const end = new Date(endDate);

    // Add 12 hours to avoid timezone issues when iterating
    current.setHours(12, 0, 0, 0);
    end.setHours(12, 0, 0, 0);

    while (current <= end) {
        queue.push(current.toISOString().split('T')[0]);
        current.setDate(current.getDate() + 1);
    }

    // Get the active tab to use for scraping
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
            bulkScrapeState = {
                isActive: true,
                queue: queue,
                total: queue.length,
                current: 0,
                results: [],
                tabId: tabs[0].id,
                retryCount: 0
            };
            processNextInQueue();
        } else {
            console.error('No active tab found for bulk scrape');
        }
    });
}

/**
 * Process the next date in the queue
 */
function processNextInQueue() {
    if (bulkScrapeState.queue.length === 0) {
        // Done!
        bulkScrapeState.isActive = false;
        chrome.runtime.sendMessage({
            action: 'BULK_SCRAPE_COMPLETE',
            data: { count: bulkScrapeState.total }
        });
        return;
    }

    // Don't shift yet! We only shift when we confirm we scraped the RIGHT date.
    const nextDate = bulkScrapeState.queue[0];

    // Notify popup of progress (using current index)
    chrome.runtime.sendMessage({
        action: 'BULK_SCRAPE_PROGRESS',
        data: {
            current: bulkScrapeState.current + 1,
            total: bulkScrapeState.total,
            date: nextDate
        }
    });

    console.log(`Processing ${nextDate}...`);

    // Tell content script to navigate
    if (bulkScrapeState.tabId) {
        chrome.tabs.sendMessage(bulkScrapeState.tabId, {
            action: 'NAVIGATE_TO_DATE',
            date: nextDate
        }).catch(err => {
            console.error("Error sending navigation message:", err);
            // If message fails (e.g. tab loading), wait and retry
            setTimeout(processNextInQueue, 1000);
        });
    }
}

/**
 * Handle newly scraped shifts notification
 */
function handleShiftsScraped(data) {
    console.log(`Background: Received ${data.shiftCount} shifts for ${data.date}`);

    if (bulkScrapeState.isActive) {
        const targetDate = bulkScrapeState.queue[0];

        // Verify if we scraped the correct date
        if (data.date === targetDate) {
            // Success! Move to next date
            console.log(`Successfully scraped target date ${targetDate}`);
            bulkScrapeState.queue.shift();
            bulkScrapeState.current++;
            bulkScrapeState.retryCount = 0;

            // Update badge
            chrome.action.setBadgeText({ text: '✓' });
            chrome.action.setBadgeBackgroundColor({ color: '#4CAF50' });
            setTimeout(() => chrome.action.setBadgeText({ text: '' }), 1000);

            // Proceed to next
            setTimeout(processNextInQueue, 1000);
        } else {
            // Wrong date - we might be on the wrong month or just arrived at a new page
            console.log(`Scraped date ${data.date} does not match target ${targetDate}. Retrying navigation...`);

            bulkScrapeState.retryCount++;
            if (bulkScrapeState.retryCount > 10) {
                console.error(`Max retries reached for ${targetDate}. Skipping.`);
                bulkScrapeState.queue.shift();
                bulkScrapeState.current++;
                bulkScrapeState.retryCount = 0;
                setTimeout(processNextInQueue, 1000);
            } else {
                // Try navigating again (this handles the Month -> Day multi-step process)
                setTimeout(processNextInQueue, 1500);
            }
        }
    } else {
        // Normal manual scrape
        chrome.action.setBadgeText({ text: '✓' });
        chrome.action.setBadgeBackgroundColor({ color: '#4CAF50' });
        setTimeout(() => chrome.action.setBadgeText({ text: '' }), 3000);
    }
}

/**
 * Get all stored shift data
 */
async function getAllShifts() {
    try {
        const result = await chrome.storage.local.get(['shiftData', 'lastScraped']);
        return {
            success: true,
            data: result.shiftData || {},
            lastScraped: result.lastScraped
        };
    } catch (error) {
        console.error('Error getting shifts:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Get user's class calendar
 */
async function getUserCalendar() {
    try {
        const result = await chrome.storage.local.get(['userCalendar']);
        return {
            success: true,
            calendar: result.userCalendar || getDefaultCalendar()
        };
    } catch (error) {
        console.error('Error getting calendar:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Save user's class calendar
 */
async function saveUserCalendar(calendar) {
    try {
        await chrome.storage.local.set({ userCalendar: calendar });
        return { success: true };
    } catch (error) {
        console.error('Error saving calendar:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Default calendar based on the provided User_Calendar.png
 * This represents the user's class schedule for Spring 2026
 * School starts January 20th, 2026 (Tuesday)
 */
function getDefaultCalendar() {
    return {
        schoolYearStart: '2026-01-20',
        classes: [
            // Monday classes
            { day: 'Monday', startTime: '09:00', endTime: '09:50', name: 'ECEG 478', location: 'Dana 216' },
            { day: 'Monday', startTime: '10:00', endTime: '10:50', name: 'ECEG 478 Lab', location: 'Dana 216' },
            { day: 'Monday', startTime: '13:00', endTime: '13:50', name: 'ECEG 310', location: 'ELC 231' },
            { day: 'Monday', startTime: '14:00', endTime: '14:50', name: 'CSCI 315', location: 'Breakiron 164' },

            // Tuesday classes  
            { day: 'Tuesday', startTime: '09:30', endTime: '10:50', name: 'ECEG 478', location: 'Dana 216' },
            { day: 'Tuesday', startTime: '11:00', endTime: '12:20', name: 'ECEG 310', location: 'ELC 231' },
            { day: 'Tuesday', startTime: '13:00', endTime: '14:50', name: 'ECEG 310 Lab', location: 'ELC 228' },

            // Wednesday classes
            { day: 'Wednesday', startTime: '09:00', endTime: '09:50', name: 'ECEG 478', location: 'Dana 216' },
            { day: 'Wednesday', startTime: '10:00', endTime: '10:50', name: 'ECEG 478 Lab', location: 'Dana 216' },
            { day: 'Wednesday', startTime: '13:00', endTime: '13:50', name: 'ECEG 310', location: 'ELC 231' },
            { day: 'Wednesday', startTime: '14:00', endTime: '14:50', name: 'CSCI 315', location: 'Breakiron 164' },

            // Thursday classes
            { day: 'Thursday', startTime: '09:30', endTime: '10:50', name: 'ECEG 478', location: 'Dana 216' },
            { day: 'Thursday', startTime: '11:00', endTime: '12:20', name: 'ECEG 310', location: 'ELC 231' },
            { day: 'Thursday', startTime: '14:00', endTime: '15:50', name: 'CSCI 315 Lab', location: 'Breakiron 164' },

            // Friday classes
            { day: 'Friday', startTime: '09:00', endTime: '09:50', name: 'ECEG 478', location: 'Dana 216' },
            { day: 'Friday', startTime: '13:00', endTime: '13:50', name: 'ECEG 310', location: 'ELC 231' },
            { day: 'Friday', startTime: '14:00', endTime: '14:50', name: 'CSCI 315', location: 'Breakiron 164' }
        ],
        // No classes on Saturday/Sunday
        holidays: [
            { date: '2026-01-19', name: 'MLK Day' }
        ]
    };
}

/**
 * Clear all stored data
 */
async function clearAllData() {
    try {
        await chrome.storage.local.clear();
        return { success: true };
    } catch (error) {
        console.error('Error clearing data:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Get statistics about stored data
 */
async function getStats() {
    try {
        const result = await chrome.storage.local.get(['shiftData', 'lastScraped', 'userCalendar']);

        const shiftData = result.shiftData || {};
        const dates = Object.keys(shiftData);
        let totalShifts = 0;
        let availableShifts = 0;
        let yourShifts = 0;

        dates.forEach(date => {
            const dayData = shiftData[date];
            if (dayData && dayData.shifts) {
                totalShifts += dayData.shifts.length;
                dayData.shifts.forEach(shift => {
                    if (shift.status === 'available' || shift.status === 'dropped') {
                        availableShifts++;
                    }
                    if (shift.status === 'your_shift') {
                        yourShifts++;
                    }
                });
            }
        });

        return {
            success: true,
            stats: {
                daysScraped: dates.length,
                totalShifts,
                availableShifts,
                yourShifts,
                lastScraped: result.lastScraped,
                hasCalendar: !!result.userCalendar
            }
        };
    } catch (error) {
        console.error('Error getting stats:', error);
        return { success: false, error: error.message };
    }
}

// Initialize on install
chrome.runtime.onInstalled.addListener(() => {
    console.log('Shift Scheduler Optimizer installed');

    // Set default calendar
    chrome.storage.local.get(['userCalendar'], (result) => {
        if (!result.userCalendar) {
            chrome.storage.local.set({ userCalendar: getDefaultCalendar() });
        }
    });
});
