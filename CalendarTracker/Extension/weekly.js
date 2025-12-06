/**
 * Weekly View Script for Shift Scheduler Optimizer
 * Displays weekly schedule grid with shifts and classes
 */

(function () {
    // Time slots from 8 AM to 1 AM (next day) - supports late night shifts
    var TIME_SLOTS = [];
    // 8 AM (8) to 11 PM (23) = hours 8-23
    // Then 12 AM (0), 1 AM (1) represented as 24, 25 for sorting
    for (var hour = 8; hour <= 25; hour++) {
        var displayHour = hour > 23 ? hour - 24 : hour;
        TIME_SLOTS.push((displayHour < 10 ? '0' : '') + displayHour + ':00');
    }
    var DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    var MAX_WEEKLY_HOURS = 20; // University rule

    // Filter state
    var filters = {
        optimal: true,
        available: true,
        dropped: true,
        your_shift: true,
        taken: true,
        class: true
    };

    // Store data globally
    var allShiftData = {};
    var userCalendar = null;
    var userConfig = {};
    var availableWeeks = [];
    var currentWeekIndex = 0;

    function loadData() {
        console.log('Weekly view: Loading data...');

        chrome.storage.local.get(['shiftData', 'userCalendar', 'userConfig'], function (result) {
            console.log('Weekly view: Storage result', result);

            allShiftData = result.shiftData || {};
            userCalendar = result.userCalendar;
            userConfig = result.userConfig || {};

            var keys = Object.keys(allShiftData);
            console.log('Weekly view: Found', keys.length, 'days of shift data');

            if (keys.length === 0) {
                console.log('Weekly view: No shift data found');
                document.getElementById('weekSelect').innerHTML = '<option value="">No data available</option>';
                return;
            }

            // Build list of available weeks from the data
            buildWeeksList(keys);

            // Populate the week selector dropdown
            populateWeekSelector();

            // Load the current week by default
            loadWeek(currentWeekIndex);

            setupFilters();
            setupWeekSelector();
        });
    }

    /**
     * Build list of weeks based on available shift data
     */
    function buildWeeksList(dateParts) {
        var weekMap = {};

        dateParts.forEach(function (dateStr) {
            var weekRange = getWeekRangeForDate(dateStr);
            var weekKey = weekRange.start;

            if (!weekMap[weekKey]) {
                weekMap[weekKey] = {
                    start: weekRange.start,
                    end: weekRange.end,
                    startDate: weekRange.startDate,
                    endDate: weekRange.endDate,
                    datesWithData: []
                };
            }
            weekMap[weekKey].datesWithData.push(dateStr);
        });

        // Convert to array and sort
        availableWeeks = Object.values(weekMap).sort(function (a, b) {
            return new Date(a.start) - new Date(b.start);
        });

        // Find current week
        var today = new Date();
        var todayStr = formatDateISO(today);

        for (var i = 0; i < availableWeeks.length; i++) {
            if (todayStr >= availableWeeks[i].start && todayStr <= availableWeeks[i].end) {
                currentWeekIndex = i;
                break;
            }
        }

        console.log('Weekly view: Found', availableWeeks.length, 'weeks of data');
    }

    /**
     * Get week range (Sunday to Saturday) for a given date string
     */
    function getWeekRangeForDate(dateStr) {
        var date = new Date(dateStr + 'T12:00:00');
        var day = date.getDay();

        var sunday = new Date(date);
        sunday.setDate(date.getDate() - day);

        var saturday = new Date(sunday);
        saturday.setDate(sunday.getDate() + 6);

        return {
            start: formatDateISO(sunday),
            end: formatDateISO(saturday),
            startDate: sunday,
            endDate: saturday
        };
    }

    function formatDateISO(date) {
        var y = date.getFullYear();
        var m = String(date.getMonth() + 1).padStart(2, '0');
        var d = String(date.getDate()).padStart(2, '0');
        return y + '-' + m + '-' + d;
    }

    function formatDateDisplay(date) {
        var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        return months[date.getMonth()] + ' ' + date.getDate();
    }

    /**
     * Populate week selector dropdown
     */
    function populateWeekSelector() {
        var select = document.getElementById('weekSelect');
        var html = '';

        var today = new Date();
        var todayStr = formatDateISO(today);

        availableWeeks.forEach(function (week, index) {
            // Create fresh date objects from ISO strings to avoid timezone issues
            var startDate = new Date(week.start + 'T12:00:00');
            var endDate = new Date(week.end + 'T12:00:00');
            var startDisplay = formatDateDisplay(startDate);
            var endDisplay = formatDateDisplay(endDate);
            var isCurrent = todayStr >= week.start && todayStr <= week.end;
            var label = startDisplay + ' - ' + endDisplay + (isCurrent ? ' (Current)' : '');

            html += '<option value="' + index + '"' + (index === currentWeekIndex ? ' selected' : '') + '>' + label + '</option>';
        });

        select.innerHTML = html || '<option value="">No weeks available</option>';
    }

    /**
     * Setup week selector change handler
     */
    function setupWeekSelector() {
        var select = document.getElementById('weekSelect');
        select.addEventListener('change', function () {
            var index = parseInt(this.value);
            if (!isNaN(index)) {
                loadWeek(index);
            }
        });
    }

    /**
     * Load and display a specific week
     */
    function loadWeek(weekIndex) {
        if (weekIndex < 0 || weekIndex >= availableWeeks.length) {
            showNoDataMessage('Week not available. Please select another week.');
            return;
        }

        currentWeekIndex = weekIndex;
        var week = availableWeeks[weekIndex];

        // Check data coverage for this week
        var coverage = checkWeekCoverage(week);
        updateWeekInfo(week, coverage);

        // Filter shift data for this week only
        var weekShiftData = {};
        Object.keys(allShiftData).forEach(function (dateStr) {
            if (dateStr >= week.start && dateStr <= week.end) {
                var dayData = allShiftData[dateStr];
                if (dayData && dayData.shifts) {
                    var filtered = Optimizer.deduplicateShifts(Optimizer.filterByRole(dayData.shifts, userConfig));
                    weekShiftData[dateStr] = Object.assign({}, dayData, { shifts: filtered });
                }
            }
        });

        // Show message if NO data at all for this week
        if (Object.keys(weekShiftData).length === 0) {
            var startDisplay = formatDateDisplay(new Date(week.start + 'T12:00:00'));
            var endDisplay = formatDateDisplay(new Date(week.end + 'T12:00:00'));
            showNoDataMessage('No shift data for this week. Visit Conportal to scrape data for ' +
                startDisplay + ' - ' + endDisplay + '.');
            return;
        }

        console.log('Weekly view: Loading week', week.start, 'to', week.end, 'with', Object.keys(weekShiftData).length, 'days');

        // Get weekly grid and render - show even partial data
        var weeklyData = Optimizer.getWeeklyGrid(weekShiftData, userCalendar, userConfig);
        console.log('Weekly view: Grid data', weeklyData);

        renderGrid(weeklyData, week);
        updateStats(weeklyData.result, week);
    }

    /**
     * Check how many days of the week have data
     */
    function checkWeekCoverage(week) {
        var daysWithData = 0;
        var missingDays = [];

        var currentDate = new Date(week.startDate);
        for (var i = 0; i < 7; i++) {
            var dateStr = formatDateISO(currentDate);
            if (allShiftData[dateStr]) {
                daysWithData++;
            } else {
                missingDays.push(DAYS[currentDate.getDay()]);
            }
            currentDate.setDate(currentDate.getDate() + 1);
        }

        return {
            daysWithData: daysWithData,
            totalDays: 7,
            missingDays: missingDays,
            isComplete: daysWithData === 7
        };
    }

    /**
     * Update week info display
     */
    function updateWeekInfo(week, coverage) {
        var infoEl = document.getElementById('weekInfo');

        if (coverage.isComplete) {
            infoEl.textContent = '✓ Complete week data';
            infoEl.className = 'week-info';
        } else {
            infoEl.textContent = '⚠ Missing: ' + coverage.missingDays.join(', ') + ' - Scrape to complete';
            infoEl.className = 'week-info warning';
        }
    }

    /**
     * Show no data message
     */
    function showNoDataMessage(message) {
        var container = document.getElementById('gridContainer');
        container.innerHTML = '<div class="no-data warning"><h2>⚠ Missing Data</h2><p>' + message + '</p></div>';
        document.getElementById('summaryBar').style.display = 'none';
    }

    // Check if a shift overlaps with any class on the same day
    function hasClassConflict(shift, dayName) {
        if (!userCalendar || !userCalendar.classes) return false;

        var shiftStart = timeToMinutes(shift.startTime);
        var shiftEnd = timeToMinutes(shift.endTime);

        for (var i = 0; i < userCalendar.classes.length; i++) {
            var cls = userCalendar.classes[i];
            if (cls.day === dayName) {
                var classStart = timeToMinutes(cls.startTime);
                var classEnd = timeToMinutes(cls.endTime);

                // Check for overlap
                if (shiftStart < classEnd && shiftEnd > classStart) {
                    return true;
                }
            }
        }
        return false;
    }

    function timeToMinutes(timeStr) {
        var parts = timeStr.split(':');
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }

    function getBlockHeight(startTime, endTime) {
        var startParts = startTime.split(':');
        var endParts = endTime.split(':');
        var startHour = parseInt(startParts[0]);
        var endHour = parseInt(endParts[0]);

        // Handle late night hours (0, 1, 2 AM treated as 24, 25, 26)
        if (startHour < 8) startHour += 24;
        if (endHour < 8) endHour += 24;
        // Also handle 11:xx as 23:xx (11 PM)
        if (startHour === 11 && parseInt(startParts[1]) >= 0) startHour = 23;
        if (endHour === 11 && parseInt(endParts[1]) >= 59) endHour = 24; // 11:59 -> 12:00 AM

        var startMin = startHour * 60 + parseInt(startParts[1]);
        var endMin = endHour * 60 + parseInt(endParts[1]);

        return ((endMin - startMin) / 60) * 40;
    }

    function getBlockTop(startTime) {
        var parts = startTime.split(':');
        var h = parseInt(parts[0]);
        var m = parseInt(parts[1]);

        // Handle late night hours (0, 1, 2 AM treated as 24, 25, 26)
        if (h < 8) h += 24;

        return (h - 8) * 40 + (m / 60) * 40;
    }

    var SEMESTER_START = '2026-01-20'; // Classes start January 20, 2026
    var SEMESTER_END = '2026-05-04';   // Classes end May 4, 2026

    function renderGrid(weeklyData, week) {
        var container = document.getElementById('gridContainer');

        // Clear any existing no-data message
        var noDataEl = document.getElementById('noData');
        if (noDataEl) noDataEl.style.display = 'none';

        // Build date headers with actual dates from the SELECTED week
        var html = '<div class="week-grid">';
        html += '<div class="grid-header time-col">Time</div>';

        // Parse the week start date correctly (add T12:00:00 to avoid timezone issues)
        var weekStartDate = new Date(week.start + 'T12:00:00');

        DAYS.forEach(function (day, idx) {
            var thisDate = new Date(weekStartDate);
            thisDate.setDate(weekStartDate.getDate() + idx);
            var dateStr = formatDateISO(thisDate);
            var hasData = !!allShiftData[dateStr];
            var dateDisplay = (thisDate.getMonth() + 1) + '/' + thisDate.getDate();
            html += '<div class="grid-header">' + day + '<br><small style="opacity:0.7">' + dateDisplay + '</small>' +
                (hasData ? '' : '<br><small style="color:#f59e0b">⚠ No data</small>') + '</div>';
        });

        TIME_SLOTS.forEach(function (time, idx) {
            var hour = parseInt(time.split(':')[0]);
            var displayTime;
            if (hour === 0) {
                displayTime = '12 AM';
            } else if (hour === 12) {
                displayTime = '12 PM';
            } else if (hour > 12) {
                displayTime = (hour - 12) + ' PM';
            } else {
                displayTime = hour + ' AM';
            }
            html += '<div class="time-slot">' + displayTime + '</div>';
            DAYS.forEach(function (day) { html += '<div class="day-column" data-day="' + day + '" data-slot="' + idx + '"></div>'; });
        });
        html += '</div>';
        container.innerHTML = html;

        // Parse week start for per-day date calculations
        var weekStartDate = new Date(week.start + 'T12:00:00');

        DAYS.forEach(function (day, dayIndex) {
            var dayData = weeklyData.grid[day];
            if (!dayData) return;

            // Calculate this specific day's date
            var thisDayDate = new Date(weekStartDate);
            thisDayDate.setDate(weekStartDate.getDate() + dayIndex);
            var thisDayStr = formatDateISO(thisDayDate);

            // Check if THIS DAY is on or after semester start (Jan 20, 2026)
            var showClassesForDay = thisDayStr >= SEMESTER_START && thisDayStr <= SEMESTER_END;

            // Only render classes if this specific day is during the semester
            if (showClassesForDay) {
                dayData.classes.forEach(function (c) {
                    addBlock(day, c.startTime, c.endTime, { type: 'class', name: c.name });
                });
            }

            // Render shifts, skipping those that conflict with classes (only if during semester)
            dayData.shifts.forEach(function (s) {
                // Skip shifts that overlap with classes (only check during semester)
                if (showClassesForDay && hasClassConflict(s, day)) {
                    return;
                }

                addBlock(day, s.startTime, s.endTime, {
                    type: 'shift', status: s.status, isOptimal: s.isOptimal, role: s.role, score: s.score
                });
            });
        });

        document.getElementById('summaryBar').style.display = 'flex';
    }

    function addBlock(day, startTime, endTime, data) {
        var startHour = parseInt(startTime.split(':')[0]);

        // Handle late night hours (0, 1, 2 AM treated as 24, 25, 26)
        var adjustedHour = startHour < 8 ? startHour + 24 : startHour;
        if (adjustedHour < 8) return; // Skip if before 8 AM

        var slotIdx = adjustedHour - 8;
        var col = document.querySelector('.day-column[data-day="' + day + '"][data-slot="' + slotIdx + '"]');
        if (!col) return;

        var block = document.createElement('div');
        var top = getBlockTop(startTime) - (slotIdx * 40);
        var height = getBlockHeight(startTime, endTime);

        if (data.type === 'class') {
            block.className = 'class-block';
            block.setAttribute('data-type', 'class');
            block.innerHTML = '<div class="shift-role">' + data.name + '</div>';
        } else {
            var cls = 'shift-block ' + data.status;
            if (data.isOptimal) cls += ' optimal';
            block.className = cls;
            block.setAttribute('data-type', data.status);
            block.setAttribute('data-optimal', data.isOptimal ? 'true' : 'false');

            var scoreHtml = '';
            if (data.score !== undefined && data.score !== null) {
                var scoreClass = data.score >= 0 ? 'positive' : 'negative';
                scoreHtml = '<span class="shift-score ' + scoreClass + '">' + (data.score >= 0 ? '+' : '') + data.score + '</span>';
            }

            block.innerHTML = '<div class="shift-role">' + data.role + '</div>' + scoreHtml;
        }

        block.style.top = top + 'px';
        block.style.height = height + 'px';
        col.appendChild(block);
    }

    function setupFilters() {
        var toggles = document.querySelectorAll('.filter-toggle');

        toggles.forEach(function (toggle) {
            var checkbox = toggle.querySelector('input');
            var filterType = toggle.getAttribute('data-filter');

            // Set initial state
            checkbox.checked = filters[filterType];
            updateToggleAppearance(toggle, checkbox.checked);

            // Handle click
            toggle.addEventListener('click', function (e) {
                e.preventDefault();
                checkbox.checked = !checkbox.checked;
                filters[filterType] = checkbox.checked;
                updateToggleAppearance(toggle, checkbox.checked);
                applyFilters();
            });
        });
    }

    function updateToggleAppearance(toggle, isActive) {
        var check = toggle.querySelector('.check');
        if (isActive) {
            toggle.classList.remove('inactive');
            check.textContent = '✓';
        } else {
            toggle.classList.add('inactive');
            check.textContent = '✗';
        }
    }

    function applyFilters() {
        // Handle shift blocks
        var shiftBlocks = document.querySelectorAll('.shift-block');
        shiftBlocks.forEach(function (block) {
            var type = block.getAttribute('data-type');
            var isOptimal = block.getAttribute('data-optimal') === 'true';

            var shouldShow = false;

            // If it's optimal and optimal filter is on, show it
            if (isOptimal && filters.optimal) {
                shouldShow = true;
            }
            // Otherwise check the status filter
            else if (!isOptimal && filters[type]) {
                shouldShow = true;
            }
            // If optimal filter is off and it's optimal, check non-optimal filter
            else if (isOptimal && !filters.optimal && filters[type]) {
                shouldShow = true;
            }

            if (shouldShow) {
                block.classList.remove('hidden');
            } else {
                block.classList.add('hidden');
            }
        });

        // Handle class blocks
        var classBlocks = document.querySelectorAll('.class-block');
        classBlocks.forEach(function (block) {
            if (filters.class) {
                block.classList.remove('hidden');
            } else {
                block.classList.add('hidden');
            }
        });
    }

    function updateStats(result, week) {
        var currentHours = result.currentHours;

        // Update header stats with 20-hour limit check
        var hoursEl = document.getElementById('currentHours');
        var hoursContainer = document.getElementById('hoursStatContainer');
        hoursEl.textContent = Math.round(currentHours);

        // Check against 20-hour limit
        if (currentHours > MAX_WEEKLY_HOURS) {
            hoursContainer.classList.add('danger');
            hoursContainer.classList.remove('warning');
        } else if (currentHours > MAX_WEEKLY_HOURS - 3) {
            hoursContainer.classList.add('warning');
            hoursContainer.classList.remove('danger');
        } else {
            hoursContainer.classList.remove('warning', 'danger');
        }

        document.getElementById('targetHours').textContent = result.targetHoursMin + '-' + result.targetHoursMax;
        document.getElementById('recommendedCount').textContent = result.optimalSetIds.length;

        var valid = result.availableShifts.filter(function (s) { return !s.hasConflict; });
        var streak = valid.filter(function (s) {
            return s.reasons && s.reasons.some(function (r) { return r.indexOf('streak') >= 0 || r.indexOf('consecutive') >= 0; });
        });

        document.getElementById('totalAvailable').textContent = valid.length;
        document.getElementById('streakShifts').textContent = streak.length;
        document.getElementById('hoursNeeded').textContent = result.hoursNeeded;
        document.getElementById('optimalShifts').textContent = result.optimalSetIds.length;

        var hi = document.getElementById('hoursNeededItem');
        if (result.hoursNeeded > 0) {
            hi.classList.add('warning');
            hi.classList.remove('success');
        } else {
            hi.classList.add('success');
            hi.classList.remove('warning');
        }

        // Check if adding recommended would exceed 20 hours
        var projectedHours = currentHours;
        result.optimalSetIds.forEach(function (id) {
            var shift = result.availableShifts.find(function (s) { return s.id === id; });
            if (shift) projectedHours += shift.durationHours;
        });

        if (projectedHours > MAX_WEEKLY_HOURS) {
            var weekInfo = document.getElementById('weekInfo');
            weekInfo.textContent = '⛔ Warning: Recommendations would exceed 20-hour limit!';
            weekInfo.className = 'week-info danger';
        }
    }

    document.addEventListener('DOMContentLoaded', loadData);
})();
