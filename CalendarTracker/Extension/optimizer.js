/**
 * Shift Optimization Engine
 * Calculates optimal work schedule based on student constraints
 */

const Optimizer = (function () {
    'use strict';

    // Default configuration (can be overridden by user settings)
    const DEFAULT_CONFIG = {
        TARGET_HOURS_MIN: 13,
        TARGET_HOURS_MAX: 15,
        MAX_WEEKLY_HOURS: 20,             // University rule - cannot exceed 20 hours/week
        PREFERRED_START_HOUR: 16, // 4:00 PM
        PREFERRED_END_HOUR: 22,   // 10:00 PM
        SCHOOL_YEAR_START: '2026-01-20', // Tuesday, January 20th
        MLK_HOLIDAY: '2026-01-19'         // Monday, January 19th
    };

    // Scoring weights
    const SCORE = {
        PREFERRED_TIME: 10,           // Shift is between preferred hours
        HELPS_TARGET: 5,              // Helps hit target hours exactly
        TWO_HOUR_STREAK: 8,           // Part of a 2+ hour consecutive block
        CONSECUTIVE_SHIFT: 4,         // Adjacent to another available shift
        CLASS_CONFLICT: -100,         // Overlaps with a class
        BEFORE_PREFERRED: -50,        // Before preferred time (unless needed)
        DROPPED_SHIFT_BONUS: 3,       // Slight preference for dropped shifts
        EXTENDS_YOUR_SHIFT: 6         // Adjacent to a shift user already has
    };

    /**
     * Parse time string "HH:MM" to hours number
     */
    function timeToHours(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        return hours + minutes / 60;
    }

    /**
     * Parse time string to minutes since midnight
     */
    function timeToMinutes(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        return hours * 60 + minutes;
    }

    /**
     * Check if two time ranges overlap
     */
    function timeRangesOverlap(start1, end1, start2, end2) {
        const s1 = timeToHours(start1);
        const e1 = timeToHours(end1);
        const s2 = timeToHours(start2);
        const e2 = timeToHours(end2);

        return s1 < e2 && s2 < e1;
    }

    /**
     * Check if two shifts are adjacent (end time of one = start time of other)
     */
    function shiftsAreAdjacent(shift1, shift2) {
        if (shift1.date !== shift2.date) return false;
        return shift1.endTime === shift2.startTime || shift2.endTime === shift1.startTime;
    }

    /**
     * Get day name from date string (YYYY-MM-DD)
     */
    function getDayName(dateStr) {
        const date = new Date(dateStr + 'T12:00:00');
        const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        return days[date.getDay()];
    }

    /**
     * Check if a shift conflicts with any class
     */
    function hasClassConflict(shift, calendar) {
        if (!calendar || !calendar.classes) return false;

        const shiftDay = shift.dayName || getDayName(shift.date);

        return calendar.classes.some(classItem => {
            if (classItem.day !== shiftDay) return false;
            return timeRangesOverlap(
                shift.startTime, shift.endTime,
                classItem.startTime, classItem.endTime
            );
        });
    }

    /**
     * Check if shift is in preferred time window
     */
    function isPreferredTime(shift, config) {
        const cfg = config || DEFAULT_CONFIG;
        return shift.startHour >= cfg.PREFERRED_START_HOUR &&
            shift.endHour <= cfg.PREFERRED_END_HOUR;
    }

    /**
     * Check if shift is before preferred time
     */
    function isBeforePreferred(shift, config) {
        const cfg = config || DEFAULT_CONFIG;
        return shift.startHour < cfg.PREFERRED_START_HOUR;
    }

    /**
     * Find consecutive shifts that could form 2+ hour blocks
     */
    function findConsecutiveShifts(shift, allShifts) {
        return allShifts.filter(other => {
            if (other.id === shift.id) return false;
            if (other.date !== shift.date) return false;
            if (other.role !== shift.role) return false;
            return shiftsAreAdjacent(shift, other);
        });
    }

    /**
     * Check if shift extends user's existing shift
     */
    function extendsUserShift(shift, yourShifts) {
        return yourShifts.some(yourShift => {
            if (yourShift.date !== shift.date) return false;
            if (yourShift.role !== shift.role) return false;
            return shiftsAreAdjacent(shift, yourShift);
        });
    }

    /**
     * Calculate score for a single shift
     */
    function scoreShift(shift, calendar, currentHours, targetRemaining, allShifts, yourShifts, config) {
        let score = 0;
        const reasons = [];
        const cfg = config || DEFAULT_CONFIG;

        // Check for class conflict (major penalty)
        if (hasClassConflict(shift, calendar)) {
            score += SCORE.CLASS_CONFLICT;
            reasons.push('Conflicts with class');
            return { score, reasons, hasConflict: true };
        }

        // Preferred time bonus
        if (isPreferredTime(shift, cfg)) {
            score += SCORE.PREFERRED_TIME;
            reasons.push('Preferred time window');
        } else if (isBeforePreferred(shift, cfg)) {
            // Only penalize if we don't need extra hours
            if (targetRemaining <= 0 || currentHours + shift.durationHours > cfg.TARGET_HOURS_MIN) {
                score += SCORE.BEFORE_PREFERRED;
                reasons.push('Before preferred time');
            }
        }

        // Check for 2+ hour streak potential
        const adjacentShifts = findConsecutiveShifts(shift, allShifts);
        const availableAdjacent = adjacentShifts.filter(s =>
            s.status === 'available' || s.status === 'dropped'
        );

        if (availableAdjacent.length > 0) {
            score += SCORE.CONSECUTIVE_SHIFT;
            reasons.push('Can form consecutive block');

            // Extra bonus if we can form 2+ hour streak
            const totalPotentialHours = shift.durationHours +
                availableAdjacent.reduce((sum, s) => sum + s.durationHours, 0);

            if (totalPotentialHours >= 2) {
                score += SCORE.TWO_HOUR_STREAK;
                reasons.push('2+ hour streak possible');
            }
        }

        // Check if extends user's existing shift
        if (extendsUserShift(shift, yourShifts)) {
            score += SCORE.EXTENDS_YOUR_SHIFT;
            reasons.push('Extends your existing shift');
        }

        // Check if helps hit target exactly
        const newTotal = currentHours + shift.durationHours;
        if (newTotal >= cfg.TARGET_HOURS_MIN && newTotal <= cfg.TARGET_HOURS_MAX) {
            score += SCORE.HELPS_TARGET;
            reasons.push('Helps reach target hours');
        } else if (newTotal > cfg.TARGET_HOURS_MAX) {
            score -= 10;
            reasons.push('Exceeds target hours');
        }

        // Bonus for dropped shifts (easier availability)
        if (shift.status === 'dropped') {
            score += SCORE.DROPPED_SHIFT_BONUS;
            reasons.push('Dropped shift available');
        }

        return { score, reasons, hasConflict: false };
    }

    /**
     * Calculate hours currently scheduled (from your_shift status) - ALL data
     */
    function calculateCurrentHours(shiftData) {
        let totalHours = 0;
        const yourShifts = [];

        Object.values(shiftData).forEach(dayData => {
            if (!dayData || !dayData.shifts) return;

            dayData.shifts.forEach(shift => {
                if (shift.status === 'your_shift') {
                    totalHours += shift.durationHours;
                    yourShifts.push(shift);
                }
            });
        });

        return { totalHours, yourShifts };
    }

    /**
     * Get current week's date range (Sunday to Saturday)
     * @param {Date} referenceDate - Optional date to calculate week for (defaults to today)
     * @returns {Object} { start: 'YYYY-MM-DD', end: 'YYYY-MM-DD', startDate: Date, endDate: Date }
     */
    function getCurrentWeekRange(referenceDate) {
        const date = referenceDate ? new Date(referenceDate) : new Date();
        const day = date.getDay(); // 0 = Sunday

        // Get Sunday of current week
        const sunday = new Date(date);
        sunday.setDate(date.getDate() - day);
        sunday.setHours(0, 0, 0, 0);

        // Get Saturday of current week
        const saturday = new Date(sunday);
        saturday.setDate(sunday.getDate() + 6);
        saturday.setHours(23, 59, 59, 999);

        return {
            start: formatDateISO(sunday),
            end: formatDateISO(saturday),
            startDate: sunday,
            endDate: saturday
        };
    }

    /**
     * Calculate hours for a specific week only (Sunday to Saturday)
     * @param {Object} shiftData - All shift data keyed by date
     * @param {Date} weekDate - Optional date to determine which week (defaults to first date in data or current)
     * @returns {Object} { weeklyHours, yourShifts, weekRange }
     */
    function calculateWeeklyHours(shiftData, weekDate) {
        // If no weekDate provided, infer from the first date in the data
        if (!weekDate) {
            const dates = Object.keys(shiftData).sort();
            if (dates.length > 0) {
                weekDate = new Date(dates[0] + 'T12:00:00');
            }
        }

        const weekRange = getCurrentWeekRange(weekDate);
        let weeklyHours = 0;
        const yourShifts = [];

        Object.keys(shiftData).forEach(dateStr => {
            // Check if this date falls within the week
            if (dateStr >= weekRange.start && dateStr <= weekRange.end) {
                const dayData = shiftData[dateStr];
                if (!dayData || !dayData.shifts) return;

                dayData.shifts.forEach(shift => {
                    if (shift.status === 'your_shift') {
                        weeklyHours += shift.durationHours;
                        yourShifts.push(shift);
                    }
                });
            }
        });

        return { weeklyHours, yourShifts, weekRange };
    }

    /**
     * Get all shifts as flat array
     */
    function getAllShiftsFlat(shiftData) {
        const all = [];
        Object.values(shiftData).forEach(dayData => {
            if (dayData && dayData.shifts) {
                all.push(...dayData.shifts);
            }
        });
        return all;
    }

    /**
     * Get all available shifts (open or dropped)
     */
    function getAvailableShifts(shiftData) {
        const available = [];

        Object.values(shiftData).forEach(dayData => {
            if (!dayData || !dayData.shifts) return;

            dayData.shifts.forEach(shift => {
                if (shift.status === 'available' || shift.status === 'dropped') {
                    available.push(shift);
                }
            });
        });

        return available;
    }

    /**
     * Main optimization function
     * Returns recommended shifts sorted by score
     */
    function optimizeSchedule(shiftData, calendar, userConfig) {
        const config = { ...DEFAULT_CONFIG, ...userConfig };
        // Use weekly hours calculation (Sunday to Saturday)
        const { weeklyHours: currentHours, yourShifts } = calculateWeeklyHours(shiftData);
        const allShifts = getAllShiftsFlat(shiftData);
        const availableShifts = getAvailableShifts(shiftData);

        const targetRemaining = config.TARGET_HOURS_MIN - currentHours;

        // Score each available shift
        const scoredShifts = availableShifts.map(shift => {
            const { score, reasons, hasConflict } = scoreShift(
                shift, calendar, currentHours, targetRemaining, allShifts, yourShifts, config
            );

            return {
                ...shift,
                score,
                reasons,
                hasConflict,
                isOptimal: score >= 10 && !hasConflict
            };
        });

        // Sort by score (highest first)
        scoredShifts.sort((a, b) => b.score - a.score);

        // Determine optimal set to reach target hours (but not exceed MAX_WEEKLY_HOURS)
        const maxAllowed = config.MAX_WEEKLY_HOURS || 20;
        const roomToAdd = maxAllowed - currentHours;
        let hoursToAdd = Math.max(0, Math.min(config.TARGET_HOURS_MIN - currentHours, roomToAdd));
        let hoursAccumulated = 0;
        const optimalSet = [];

        for (const shift of scoredShifts) {
            if (shift.hasConflict) continue;
            if (hoursAccumulated >= hoursToAdd) break;

            // Don't recommend if it would exceed 20-hour limit
            if (currentHours + hoursAccumulated + shift.durationHours > maxAllowed) continue;

            optimalSet.push(shift.id);
            hoursAccumulated += shift.durationHours;
        }

        return {
            currentHours,
            maxWeeklyHours: maxAllowed,
            targetHoursMin: config.TARGET_HOURS_MIN,
            targetHoursMax: config.TARGET_HOURS_MAX,
            hoursNeeded: Math.max(0, config.TARGET_HOURS_MIN - currentHours),
            hoursRemaining: Math.max(0, maxAllowed - currentHours),
            yourShifts,
            allShifts,
            availableShifts: scoredShifts,
            optimalSetIds: optimalSet,
            config,
            summary: generateSummary(currentHours, yourShifts, scoredShifts, optimalSet, config)
        };
    }

    /**
     * Generate human-readable summary
     */
    function generateSummary(currentHours, yourShifts, scoredShifts, optimalSet, config) {
        const lines = [];

        if (currentHours >= config.TARGET_HOURS_MIN) {
            if (currentHours <= config.TARGET_HOURS_MAX) {
                lines.push(`✅ You have ${currentHours} hours scheduled (target: ${config.TARGET_HOURS_MIN}-${config.TARGET_HOURS_MAX}h)`);
            } else {
                lines.push(`⚠️ You have ${currentHours} hours scheduled (exceeds target of ${config.TARGET_HOURS_MAX}h)`);
            }
        } else {
            const needed = config.TARGET_HOURS_MIN - currentHours;
            lines.push(`📊 You have ${currentHours} hours, need ${needed} more to reach minimum target`);
        }

        const validShifts = scoredShifts.filter(s => !s.hasConflict);
        const preferredShifts = validShifts.filter(s => s.score >= 10);
        const streakShifts = validShifts.filter(s => s.reasons.some(r => r.includes('streak')));

        lines.push(`🔍 Found ${validShifts.length} available shifts (${preferredShifts.length} in preferred time)`);

        if (streakShifts.length > 0) {
            lines.push(`⏱️ ${streakShifts.length} shifts can form 2+ hour blocks`);
        }

        if (optimalSet.length > 0) {
            lines.push(`⭐ ${optimalSet.length} optimal shifts recommended`);
        }

        return lines.join('\n');
    }

    /**
     * Check if a specific date is the MLK Holiday
     */
    function isMLKHoliday(dateStr) {
        return dateStr === DEFAULT_CONFIG.MLK_HOLIDAY;
    }

    /**
     * Get weekly grid data for full-tab view
     */
    function getWeeklyGrid(shiftData, calendar, userConfig) {
        const config = { ...DEFAULT_CONFIG, ...userConfig };
        const result = optimizeSchedule(shiftData, calendar, userConfig);

        // Organize by day and time
        const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        const grid = {};

        days.forEach(day => {
            grid[day] = {
                shifts: [],
                classes: calendar?.classes?.filter(c => c.day === day) || []
            };
        });

        // Add all shifts to grid
        result.allShifts.forEach(shift => {
            const day = shift.dayName || getDayName(shift.date);
            if (grid[day]) {
                const scored = result.availableShifts.find(s => s.id === shift.id);
                grid[day].shifts.push({
                    ...shift,
                    score: scored?.score,
                    reasons: scored?.reasons || [],
                    isOptimal: result.optimalSetIds.includes(shift.id)
                });
            }
        });

        // Sort shifts by time
        days.forEach(day => {
            grid[day].shifts.sort((a, b) => timeToMinutes(a.startTime) - timeToMinutes(b.startTime));
        });

        return {
            grid,
            days,
            result,
            config
        };
    }

    /**
     * Filter shifts by role settings
     * Always excludes Stacks, optionally includes Leader
     */
    function filterByRole(shifts, config) {
        const cfg = config || {};
        return shifts.filter(shift => {
            // Always exclude Stacks
            if (shift.role === 'Stacks') return false;
            // Optionally exclude Leader
            if (shift.role === 'Leader' && !cfg.INCLUDE_LEADER) return false;
            return true;
        });
    }

    /**
     * Remove duplicate shifts by ID
     */
    function deduplicateShifts(shifts) {
        const seen = new Set();
        return shifts.filter(shift => {
            if (seen.has(shift.id)) return false;
            seen.add(shift.id);
            return true;
        });
    }

    // =============================================
    // EARNINGS CALCULATOR
    // =============================================

    const EARNINGS_CONFIG = {
        HOURLY_RATE: 10.25,
        TAX_STATE: 0.0307,      // PA State: 3.07%
        TAX_CITY: 0.0107,       // City: 1.07%
        TAX_LST: 0.0065,        // PA LST: ~0.65%
        WORK_PERIOD_START: '2026-01-18', // Sunday
        WORK_PERIOD_END: '2026-05-04',   // Monday
        PAY_PERIOD_DAYS: 14              // Bi-weekly
    };

    /**
     * Get all bi-weekly pay periods for the semester
     */
    function getPayPeriods() {
        const periods = [];
        let start = new Date(EARNINGS_CONFIG.WORK_PERIOD_START + 'T00:00:00');
        const end = new Date(EARNINGS_CONFIG.WORK_PERIOD_END + 'T00:00:00');

        while (start < end) {
            const periodEnd = new Date(start);
            periodEnd.setDate(periodEnd.getDate() + 13); // 14 days total (0-13)

            // If period end goes past work end, cap it
            const actualEnd = periodEnd > end ? end : periodEnd;

            // Paycheck is Friday after period ends (next Friday after Saturday)
            const paycheck = new Date(actualEnd);
            paycheck.setDate(paycheck.getDate() + (5 - paycheck.getDay() + 7) % 7 + (paycheck.getDay() <= 5 ? 7 : 0));
            // Simplified: Friday after the Saturday = 6 days after Saturday
            const paycheckDate = new Date(actualEnd);
            paycheckDate.setDate(paycheckDate.getDate() + (6 - paycheckDate.getDay() + 7) % 7);
            if (paycheckDate <= actualEnd) paycheckDate.setDate(paycheckDate.getDate() + 7);

            periods.push({
                startDate: formatDateISO(start),
                endDate: formatDateISO(actualEnd),
                paycheckDate: formatDateISO(paycheckDate),
                startDisplay: formatDateShort(start),
                endDisplay: formatDateShort(actualEnd),
                paycheckDisplay: formatDateShort(paycheckDate)
            });

            // Move to next period
            start = new Date(actualEnd);
            start.setDate(start.getDate() + 1);
        }

        return periods;
    }

    function formatDateISO(date) {
        return date.toISOString().split('T')[0];
    }

    function formatDateShort(date) {
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }

    /**
     * Check if a date falls within a pay period
     */
    function isDateInPeriod(dateStr, period) {
        return dateStr >= period.startDate && dateStr <= period.endDate;
    }

    /**
     * Calculate earnings for a given number of hours
     */
    function calculateEarnings(hours) {
        const gross = hours * EARNINGS_CONFIG.HOURLY_RATE;
        const stateTax = gross * EARNINGS_CONFIG.TAX_STATE;
        const cityTax = gross * EARNINGS_CONFIG.TAX_CITY;
        const lstTax = gross * EARNINGS_CONFIG.TAX_LST;
        const totalTax = stateTax + cityTax + lstTax;
        const net = gross - totalTax;

        return {
            hours,
            gross: Math.round(gross * 100) / 100,
            stateTax: Math.round(stateTax * 100) / 100,
            cityTax: Math.round(cityTax * 100) / 100,
            lstTax: Math.round(lstTax * 100) / 100,
            totalTax: Math.round(totalTax * 100) / 100,
            net: Math.round(net * 100) / 100,
            hourlyRate: EARNINGS_CONFIG.HOURLY_RATE,
            effectiveRate: Math.round((net / hours) * 100) / 100
        };
    }

    /**
     * Calculate earnings by pay period from shift data
     */
    function calculateEarningsByPeriod(shiftData) {
        const periods = getPayPeriods();
        const results = [];

        periods.forEach(period => {
            let hoursInPeriod = 0;

            // Find all your_shift entries in this period
            Object.keys(shiftData).forEach(dateStr => {
                if (isDateInPeriod(dateStr, period)) {
                    const dayData = shiftData[dateStr];
                    if (dayData && dayData.shifts) {
                        dayData.shifts.forEach(shift => {
                            if (shift.status === 'your_shift') {
                                hoursInPeriod += shift.durationHours || 0;
                            }
                        });
                    }
                }
            });

            const earnings = calculateEarnings(hoursInPeriod);

            results.push({
                ...period,
                ...earnings,
                isEmpty: hoursInPeriod === 0
            });
        });

        // Calculate totals
        const totals = results.reduce((acc, p) => {
            acc.hours += p.hours;
            acc.gross += p.gross;
            acc.net += p.net;
            acc.totalTax += p.totalTax;
            return acc;
        }, { hours: 0, gross: 0, net: 0, totalTax: 0 });

        return {
            periods: results,
            totals: {
                hours: totals.hours,
                gross: Math.round(totals.gross * 100) / 100,
                net: Math.round(totals.net * 100) / 100,
                totalTax: Math.round(totals.totalTax * 100) / 100
            },
            config: EARNINGS_CONFIG
        };
    }

    /**
     * Get current pay period info
     */
    function getCurrentPayPeriod() {
        const periods = getPayPeriods();
        const today = new Date().toISOString().split('T')[0];

        for (const period of periods) {
            if (today >= period.startDate && today <= period.endDate) {
                return { ...period, isCurrent: true };
            }
        }

        // If before first period or after last
        if (today < periods[0].startDate) {
            return { ...periods[0], isCurrent: false, isUpcoming: true };
        }

        return { ...periods[periods.length - 1], isCurrent: false, isPast: true };
    }

    // Public API
    return {
        DEFAULT_CONFIG,
        SCORE,
        EARNINGS_CONFIG,
        optimizeSchedule,
        calculateCurrentHours,
        calculateWeeklyHours,
        getCurrentWeekRange,
        getAvailableShifts,
        getAllShiftsFlat,
        scoreShift,
        hasClassConflict,
        isPreferredTime,
        isMLKHoliday,
        getWeeklyGrid,
        findConsecutiveShifts,
        shiftsAreAdjacent,
        filterByRole,
        deduplicateShifts,
        // Earnings
        getPayPeriods,
        calculateEarnings,
        calculateEarningsByPeriod,
        getCurrentPayPeriod
    };
})();

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.Optimizer = Optimizer;
}
