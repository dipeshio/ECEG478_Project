/**
 * Content Script for Shift Scheduler Optimizer
 * Scrapes shift data from Bucknell Conportal shift schedule page
 */

(function () {
  'use strict';

  // Shift status types
  const ShiftStatus = {
    AVAILABLE: 'available',
    TAKEN: 'taken',
    YOUR_SHIFT: 'your_shift',
    DROPPED: 'dropped',
    UNAVAILABLE: 'unavailable'
  };

  /**
   * Parse the current date from the page header
   * Example: "Shifts for: Tuesday, January 20th, 2026."
   */
  function parseCurrentDate() {
    const dateElement = document.getElementById('shifts_by_day_date');
    if (!dateElement) return null;

    const text = dateElement.textContent.trim();
    // Match: "Day, Month DDth, YYYY"
    const match = text.match(/(\w+),\s+(\w+)\s+(\d+)(?:st|nd|rd|th),\s+(\d+)/);
    if (!match) return null;

    const [, dayName, monthName, day, year] = match;
    const months = {
      'January': 0, 'February': 1, 'March': 2, 'April': 3,
      'May': 4, 'June': 5, 'July': 6, 'August': 7,
      'September': 8, 'October': 9, 'November': 10, 'December': 11
    };

    return {
      dayName,
      date: new Date(parseInt(year), months[monthName], parseInt(day)),
      dateString: `${year}-${String(months[monthName] + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`,
      displayDate: `${monthName} ${day}, ${year}`
    };
  }

  /**
   * Parse time string to 24-hour format
   * Example: "4:00 PM" -> 16, "12:00 AM" -> 0
   */
  function parseTime(timeStr) {
    const match = timeStr.trim().match(/(\d{1,2}):(\d{2})\s*(AM|PM)/i);
    if (!match) return null;

    let [, hours, minutes, period] = match;
    hours = parseInt(hours);
    minutes = parseInt(minutes);

    if (period.toUpperCase() === 'PM' && hours !== 12) {
      hours += 12;
    } else if (period.toUpperCase() === 'AM' && hours === 12) {
      hours = 0;
    }

    return { hours, minutes, totalMinutes: hours * 60 + minutes };
  }

  /**
   * Determine shift status from CSS class
   */
  function getShiftStatus(element) {
    const classList = element.classList;

    if (classList.contains('YourPermShift') || classList.contains('YourTempShift')) {
      return ShiftStatus.YOUR_SHIFT;
    }
    if (classList.contains('OpenShift')) {
      // Check if it's a dropped shift by looking for "Dropped by:" text
      if (element.textContent.includes('Dropped by:')) {
        return ShiftStatus.DROPPED;
      }
      return ShiftStatus.AVAILABLE;
    }
    if (classList.contains('PermTakenShift') || classList.contains('TempTakenShift')) {
      return ShiftStatus.TAKEN;
    }
    if (classList.contains('DroppedShift')) {
      return ShiftStatus.DROPPED;
    }

    return ShiftStatus.UNAVAILABLE;
  }

  /**
   * Parse shift content to extract time, date range, and assignee
   */
  function parseShiftContent(element) {
    const text = element.innerHTML;
    const lines = text.split('<br>').map(line => line.replace(/<[^>]*>/g, '').trim());

    if (lines.length < 3) return null;

    // Line 0: Time range (e.g., "4:00 PM - 5:00 PM")
    const timeMatch = lines[0].match(/(.+?)\s*-\s*(.+)/);
    if (!timeMatch) return null;

    const startTime = parseTime(timeMatch[1]);
    const endTime = parseTime(timeMatch[2]);

    if (!startTime || !endTime) return null;

    // Line 1: Date range (e.g., "1/20 - 4/28")
    const dateRange = lines[1];

    // Line 2: Assignee or status
    const assignee = lines[2];

    // Calculate duration in hours
    let durationMinutes = endTime.totalMinutes - startTime.totalMinutes;
    if (durationMinutes < 0) durationMinutes += 24 * 60; // Handle overnight shifts
    const durationHours = durationMinutes / 60;

    return {
      startTime: `${String(startTime.hours).padStart(2, '0')}:${String(startTime.minutes).padStart(2, '0')}`,
      endTime: `${String(endTime.hours).padStart(2, '0')}:${String(endTime.minutes).padStart(2, '0')}`,
      startHour: startTime.hours,
      endHour: endTime.hours,
      dateRange,
      assignee,
      durationHours
    };
  }

  /**
   * Get role/location from column header
   */
  function getRoleFromColumn(table, columnIndex) {
    const headerRow = table.querySelector('tr.shiftHeader');
    if (!headerRow) return 'Unknown';

    const cells = headerRow.querySelectorAll('td');
    // Account for time column at index 0
    if (columnIndex < cells.length) {
      return cells[columnIndex].textContent.trim();
    }
    return 'Unknown';
  }

  /**
   * Scrape all shifts from the page
   */
  function scrapeShifts() {
    const currentDate = parseCurrentDate();
    if (!currentDate) {
      console.error('Could not parse current date');
      return null;
    }

    const shifts = [];
    const table = document.querySelector('#shifts_by_day table');
    if (!table) {
      console.error('Could not find shifts table');
      return null;
    }

    // Get all data rows (not header rows)
    const dataRows = table.querySelectorAll('tr:not(.shiftHeader)');

    dataRows.forEach(row => {
      const cells = row.querySelectorAll('td');

      // Skip time column (index 0) and process shift columns
      cells.forEach((cell, cellIndex) => {
        if (cellIndex === 0) return; // Skip time column

        const role = getRoleFromColumn(table, cellIndex);

        // Find all shift divs in this cell
        const shiftDivs = cell.querySelectorAll('.OpenShift, .PermTakenShift, .TempTakenShift, .YourPermShift, .YourTempShift, .DroppedShift');

        shiftDivs.forEach(shiftDiv => {
          const status = getShiftStatus(shiftDiv);
          const content = parseShiftContent(shiftDiv);

          if (content) {
            const shift = {
              id: `${currentDate.dateString}_${content.startTime}_${role}`.replace(/[:\s]/g, '_'),
              date: currentDate.dateString,
              dayName: currentDate.dayName,
              displayDate: currentDate.displayDate,
              startTime: content.startTime,
              endTime: content.endTime,
              startHour: content.startHour,
              endHour: content.endHour,
              role,
              status,
              assignee: content.assignee,
              durationHours: content.durationHours,
              dateRange: content.dateRange,
              scrapedAt: new Date().toISOString()
            };

            shifts.push(shift);
          }
        });
      });
    });

    return {
      currentDate,
      shifts,
      scrapedAt: new Date().toISOString()
    };
  }

  /**
   * Parse shift draw information from the page
   */
  function parseShiftDrawInfo() {
    const infoElement = document.getElementById('shift_draw_message');
    if (!infoElement) return null;

    const text = infoElement.textContent;

    // Extract weekly hours: "You currently have drawn X out of Y available weekly hours"
    const weeklyMatch = text.match(/(\d+)\s*out of\s*(\d+)\s*available weekly hours/);
    // Extract daily hours: "X out of Y available daily hours"
    const dailyMatch = text.match(/(\d+)\s*out of\s*(\d+)\s*available daily hours/);

    return {
      weeklyHoursUsed: weeklyMatch ? parseInt(weeklyMatch[1]) : 0,
      weeklyHoursAvailable: weeklyMatch ? parseInt(weeklyMatch[2]) : 0,
      dailyHoursUsed: dailyMatch ? parseInt(dailyMatch[1]) : 0,
      dailyHoursAvailable: dailyMatch ? parseInt(dailyMatch[2]) : 0
    };
  }

  /**
   * Save scraped data to chrome.storage.local
   */
  async function saveShiftData(data) {
    try {
      // Get existing data
      const result = await chrome.storage.local.get(['shiftData', 'lastScraped']);
      const existingData = result.shiftData || {};

      // Merge new shifts with existing data (keyed by date)
      existingData[data.currentDate.dateString] = {
        shifts: data.shifts,
        dayName: data.currentDate.dayName,
        displayDate: data.currentDate.displayDate,
        shiftDrawInfo: parseShiftDrawInfo(),
        scrapedAt: data.scrapedAt
      };

      // Save updated data
      await chrome.storage.local.set({
        shiftData: existingData,
        lastScraped: data.scrapedAt
      });

      console.log(`Saved ${data.shifts.length} shifts for ${data.currentDate.displayDate}`);
      return true;
    } catch (error) {
      console.error('Error saving shift data:', error);
      return false;
    }
  }

  /**
   * Check if the current date is January 19th (MLK Holiday - unavailable)
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

    if (message.action === 'GET_PAGE_DATA') {
      const data = scrapeShifts();
      sendResponse(data);
      return true;
    }
  });
})();
