/**
 * Popup Script for Shift Scheduler Optimizer
 * Handles UI logic, data display, settings, and optimization
 */

(function () {
    'use strict';

    // State
    let shiftData = {};
    let userCalendar = null;
    let userConfig = {};
    let optimizationResult = null;

    // Default config
    const DEFAULT_CONFIG = {
        TARGET_HOURS_MIN: 13,
        TARGET_HOURS_MAX: 15,
        PREFERRED_START_HOUR: 16,
        PREFERRED_END_HOUR: 22,
        INCLUDE_LEADER: false,
        EXCLUDE_STACKS: true
    };

    // DOM Elements
    const elements = {
        currentHours: document.getElementById('currentHours'),
        targetHours: document.getElementById('targetHours'),
        availableShifts: document.getElementById('availableShifts'),
        optimalCount: document.getElementById('optimalCount'),
        progressFill: document.getElementById('progressFill'),
        progressPercent: document.getElementById('progressPercent'),
        progressTarget: document.getElementById('progressTarget'),
        statusBadge: document.getElementById('statusBadge'),
        shiftsBody: document.getElementById('shiftsBody'),
        recommendedBody: document.getElementById('recommendedBody'),
        yourShiftsBody: document.getElementById('yourShiftsBody'),
        dayFilter: document.getElementById('dayFilter'),
        statusFilter: document.getElementById('statusFilter'),
        targetMin: document.getElementById('targetMin'),
        targetMax: document.getElementById('targetMax'),
        preferredStart: document.getElementById('preferredStart'),
        preferredEnd: document.getElementById('preferredEnd'),
        includeLeader: document.getElementById('includeLeader'),
        classList: document.getElementById('classList'),
        newClassDay: document.getElementById('newClassDay'),
        newClassName: document.getElementById('newClassName'),
        newClassStart: document.getElementById('newClassStart'),
        newClassEnd: document.getElementById('newClassEnd'),
        lastScraped: document.getElementById('lastScraped'),
        daysLoaded: document.getElementById('daysLoaded'),
        recommendationSummary: document.getElementById('recommendationSummary'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        refreshBtn: document.getElementById('refreshBtn'),
        clearDataBtn: document.getElementById('clearDataBtn'),
        exportDataBtn: document.getElementById('exportDataBtn'),
        savePrefsBtn: document.getElementById('savePrefsBtn'),
        addClassBtn: document.getElementById('addClassBtn'),
        openWeeklyBtn: document.getElementById('openWeeklyBtn'),
        toast: document.getElementById('toast'),
        toastMessage: document.getElementById('toastMessage'),
        earningsBody: document.getElementById('earningsBody'),
        semesterGross: document.getElementById('semesterGross'),
        semesterNet: document.getElementById('semesterNet'),
        recStartDate: document.getElementById('recStartDate'),
        recEndDate: document.getElementById('recEndDate'),
        applyDateRange: document.getElementById('applyDateRange'),
        bulkStartDate: document.getElementById('bulkStartDate'),
        bulkEndDate: document.getElementById('bulkEndDate'),
        startBulkScrapeBtn: document.getElementById('startBulkScrapeBtn'),
        bulkScrapeProgress: document.getElementById('bulkScrapeProgress'),
        bulkScrapeStatus: document.getElementById('bulkScrapeStatus'),
        bulkScrapePercent: document.getElementById('bulkScrapePercent'),
        bulkScrapeFill: document.getElementById('bulkScrapeFill')
    };

    function init() {
        setupTabNavigation();
        setupEventListeners();
        loadData();
    }

    function setupTabNavigation() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                const tabId = btn.dataset.tab + '-tab';
                document.getElementById(tabId).classList.add('active');
            });
        });
    }

    function setupEventListeners() {
        elements.analyzeBtn.addEventListener('click', runAnalysis);
        elements.refreshBtn.addEventListener('click', loadData);
        elements.clearDataBtn.addEventListener('click', clearData);
        elements.exportDataBtn.addEventListener('click', exportData);
        elements.savePrefsBtn.addEventListener('click', savePreferences);
        elements.addClassBtn.addEventListener('click', addClass);
        elements.openWeeklyBtn.addEventListener('click', openWeeklyView);
        elements.dayFilter.addEventListener('change', filterShifts);
        elements.statusFilter.addEventListener('change', filterShifts);
        if (elements.applyDateRange) {
            elements.applyDateRange.addEventListener('click', renderRecommendedTable);
        }
        if (elements.startBulkScrapeBtn) {
            elements.startBulkScrapeBtn.addEventListener('click', startBulkScrape);
        }

        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            if (message.action === 'BULK_SCRAPE_PROGRESS') {
                updateBulkProgress(message.data);
            } else if (message.action === 'BULK_SCRAPE_COMPLETE') {
                finishBulkScrape(message.data);
            }
        });
    }

    async function loadData() {
        try {
            updateStatus('Loading...');
            const result = await chrome.storage.local.get(['shiftData', 'userCalendar', 'userConfig', 'lastScraped']);

            shiftData = result.shiftData || {};
            userCalendar = result.userCalendar || { classes: [] };
            userConfig = { ...DEFAULT_CONFIG, ...result.userConfig };

            loadSettingsUI();
            updateDashboard();
            renderAllShiftsTable();
            renderYourShiftsTable();
            renderEarningsTable();
            renderClassList();
            updateSettingsInfo({
                lastScraped: result.lastScraped,
                daysScraped: Object.keys(shiftData).length
            });
            updateStatus('Ready');
        } catch (error) {
            console.error('Failed to load data:', error);
            updateStatus('Error');
        }
    }

    function loadSettingsUI() {
        elements.targetMin.value = userConfig.TARGET_HOURS_MIN;
        elements.targetMax.value = userConfig.TARGET_HOURS_MAX;
        elements.preferredStart.value = userConfig.PREFERRED_START_HOUR;
        elements.preferredEnd.value = userConfig.PREFERRED_END_HOUR;
        elements.includeLeader.checked = userConfig.INCLUDE_LEADER;
    }

    async function savePreferences() {
        userConfig.TARGET_HOURS_MIN = parseInt(elements.targetMin.value);
        userConfig.TARGET_HOURS_MAX = parseInt(elements.targetMax.value);
        userConfig.PREFERRED_START_HOUR = parseInt(elements.preferredStart.value);
        userConfig.PREFERRED_END_HOUR = parseInt(elements.preferredEnd.value);
        userConfig.INCLUDE_LEADER = elements.includeLeader.checked;

        await chrome.storage.local.set({ userConfig });
        showToast('Preferences saved');
        updateDashboard();
    }

    async function addClass() {
        const day = elements.newClassDay.value;
        const name = elements.newClassName.value.trim();
        const startTime = elements.newClassStart.value;
        const endTime = elements.newClassEnd.value;

        if (!name) {
            showToast('Please enter a class name');
            return;
        }

        userCalendar.classes.push({ day, name, startTime, endTime });
        await sendMessage({ action: 'SAVE_USER_CALENDAR', calendar: userCalendar });
        elements.newClassName.value = '';
        renderClassList();
        showToast('Class added');
    }

    async function removeClass(index) {
        userCalendar.classes.splice(index, 1);
        await sendMessage({ action: 'SAVE_USER_CALENDAR', calendar: userCalendar });
        renderClassList();
        showToast('Class removed');
    }

    function openWeeklyView() {
        chrome.tabs.create({ url: chrome.runtime.getURL('weekly.html') });
    }

    function filterByRole(shifts) {
        return shifts.filter(shift => {
            if (shift.role === 'Stacks') return false;
            if (shift.role === 'Leader' && !userConfig.INCLUDE_LEADER) return false;
            return true;
        });
    }

    function deduplicateShifts(shifts) {
        const seen = new Set();
        return shifts.filter(shift => {
            if (seen.has(shift.id)) return false;
            seen.add(shift.id);
            return true;
        });
    }

    function runAnalysis() {
        if (Object.keys(shiftData).length === 0) {
            showToast('No shift data available. Visit Conportal to scrape shifts.');
            return;
        }

        try {
            const filteredShiftData = {};
            Object.keys(shiftData).forEach(date => {
                const dayData = shiftData[date];
                if (dayData && dayData.shifts) {
                    filteredShiftData[date] = {
                        ...dayData,
                        shifts: deduplicateShifts(filterByRole(dayData.shifts))
                    };
                }
            });

            optimizationResult = Optimizer.optimizeSchedule(filteredShiftData, userCalendar, userConfig);
            updateDashboard();
            renderRecommendedTable();
            renderAllShiftsTable();
            document.querySelector('[data-tab="recommended"]').click();
            elements.recommendationSummary.innerHTML = '<p>' + optimizationResult.summary.replace(/\n/g, '</p><p>') + '</p>';
            showToast('Analysis complete!');
        } catch (error) {
            console.error('Optimization error:', error);
            showToast('Analysis failed: ' + error.message);
        }
    }

    function updateDashboard() {
        let currentHours = 0;
        let available = 0;
        let optimal = 0;

        if (optimizationResult) {
            const weeklyData = Optimizer.calculateWeeklyHours(shiftData);
            currentHours = weeklyData.weeklyHours;
            available = optimizationResult.availableShifts.filter(s => !s.hasConflict).length;
            optimal = optimizationResult.optimalSetIds.length;
        } else {
            const weeklyData = Optimizer.calculateWeeklyHours(shiftData);
            currentHours = weeklyData.weeklyHours;
            const allShifts = getAllShiftsFlat();
            const filtered = deduplicateShifts(filterByRole(allShifts));
            filtered.forEach(shift => {
                if (shift.status === 'available' || shift.status === 'dropped') available++;
            });
        }

        elements.currentHours.textContent = currentHours.toFixed(0);
        elements.targetHours.textContent = userConfig.TARGET_HOURS_MIN + '-' + userConfig.TARGET_HOURS_MAX;
        elements.availableShifts.textContent = available;
        elements.optimalCount.textContent = optimal;

        const targetMax = userConfig.TARGET_HOURS_MAX || 15;
        const targetMin = userConfig.TARGET_HOURS_MIN || 13;
        const progressPercent = Math.min((currentHours / targetMax) * 100, 100);

        elements.progressFill.style.width = progressPercent + '%';
        elements.progressPercent.textContent = Math.round(progressPercent) + '% of target';

        if (currentHours >= targetMin && currentHours <= targetMax) {
            elements.progressFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
        } else if (currentHours > targetMax) {
            elements.progressFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
        } else {
            elements.progressFill.style.background = 'linear-gradient(90deg, #3b82f6, #2563eb)';
        }
    }

    function getAllShiftsFlat() {
        const all = [];
        Object.values(shiftData).forEach(dayData => {
            if (dayData && dayData.shifts) all.push(...dayData.shifts);
        });
        return all;
    }

    function renderAllShiftsTable() {
        let allShifts = deduplicateShifts(filterByRole(getAllShiftsFlat()));
        allShifts.sort((a, b) => {
            const dateCompare = a.date.localeCompare(b.date);
            if (dateCompare !== 0) return dateCompare;
            return a.startTime.localeCompare(b.startTime);
        });

        if (allShifts.length === 0) {
            elements.shiftsBody.innerHTML = '<tr class="empty-row"><td colspan="6">No shifts loaded. Visit the Conportal shift schedule to scrape data.</td></tr>';
            return;
        }

        const optimalIds = optimizationResult ? optimizationResult.optimalSetIds : [];
        let html = '';
        allShifts.forEach(shift => {
            const isOptimal = optimalIds.includes(shift.id);
            const scoreInfo = optimizationResult?.availableShifts.find(s => s.id === shift.id);
            const score = scoreInfo ? scoreInfo.score : '';
            const scoreHtml = score !== '' ? '<span class="score-badge ' + (score >= 0 ? 'positive' : 'negative') + '">' + (score >= 0 ? '+' : '') + score + '</span>' : '-';

            html += '<tr class="' + (isOptimal ? 'optimal' : '') + '" data-day="' + shift.dayName + '" data-status="' + shift.status + '">';
            html += '<td>' + shift.dayName + '</td>';
            html += '<td>' + formatDate(shift.date) + '</td>';
            html += '<td>' + formatTime(shift.startTime) + ' - ' + formatTime(shift.endTime) + '</td>';
            html += '<td>' + shift.role + '</td>';
            html += '<td><span class="status-pill ' + shift.status + '">' + formatStatus(shift.status) + '</span></td>';
            html += '<td>' + scoreHtml + '</td>';
            html += '</tr>';
        });
        elements.shiftsBody.innerHTML = html;
    }

    function renderRecommendedTable() {
        if (!optimizationResult || optimizationResult.availableShifts.length === 0) {
            elements.recommendedBody.innerHTML = '<tr class="empty-row"><td colspan="6">No recommendations available. Run analysis first.</td></tr>';
            return;
        }

        const startDateStr = elements.recStartDate ? elements.recStartDate.value : '2026-01-20';
        const endDateStr = elements.recEndDate ? elements.recEndDate.value : '2026-05-04';

        let recommended = optimizationResult.availableShifts.filter(s => {
            if (s.hasConflict) return false;
            if (s.date) return s.date >= startDateStr && s.date <= endDateStr;
            return true;
        });

        recommended.sort((a, b) => {
            if (a.date && b.date && a.date !== b.date) return a.date.localeCompare(b.date);
            return b.score - a.score;
        });

        recommended = recommended.slice(0, 30);

        if (recommended.length === 0) {
            elements.recommendedBody.innerHTML = '<tr class="empty-row"><td colspan="6">No shifts found in the selected date range.</td></tr>';
            return;
        }

        let html = '';
        recommended.forEach(shift => {
            const isOptimal = optimizationResult.optimalSetIds.includes(shift.id);
            const dateDisplay = shift.date ? formatDate(shift.date) : '-';
            const reasons = shift.reasons.map(r => '<li>' + r + '</li>').join('');
            const scoreClass = shift.score >= 0 ? 'positive' : 'negative';
            const scoreSign = shift.score >= 0 ? '+' : '';

            html += '<tr class="' + (isOptimal ? 'optimal' : '') + '">';
            html += '<td>' + dateDisplay + '</td>';
            html += '<td>' + shift.dayName + '</td>';
            html += '<td>' + formatTime(shift.startTime) + ' - ' + formatTime(shift.endTime) + '</td>';
            html += '<td>' + shift.role + '</td>';
            html += '<td><span class="score-badge ' + scoreClass + '">' + scoreSign + shift.score + '</span></td>';
            html += '<td><div class="reasons-list"><ul>' + reasons + '</ul></div></td>';
            html += '</tr>';
        });
        elements.recommendedBody.innerHTML = html;
    }

    function renderYourShiftsTable() {
        let yourShifts = deduplicateShifts(filterByRole(getAllShiftsFlat().filter(s => s.status === 'your_shift')));
        yourShifts.sort((a, b) => {
            const dateCompare = a.date.localeCompare(b.date);
            if (dateCompare !== 0) return dateCompare;
            return a.startTime.localeCompare(b.startTime);
        });

        if (yourShifts.length === 0) {
            elements.yourShiftsBody.innerHTML = '<tr class="empty-row"><td colspan="5">No scheduled shifts found.</td></tr>';
            return;
        }

        let html = '';
        yourShifts.forEach(shift => {
            html += '<tr>';
            html += '<td>' + shift.dayName + '</td>';
            html += '<td>' + formatDate(shift.date) + '</td>';
            html += '<td>' + formatTime(shift.startTime) + ' - ' + formatTime(shift.endTime) + '</td>';
            html += '<td>' + shift.role + '</td>';
            html += '<td>' + shift.durationHours + 'h</td>';
            html += '</tr>';
        });
        elements.yourShiftsBody.innerHTML = html;
    }

    function renderEarningsTable() {
        const payPeriods = Optimizer.getPayPeriods();
        const earningsData = Optimizer.calculateEarningsByPeriod(shiftData);

        if (!earningsData || !earningsData.periods || payPeriods.length === 0) {
            elements.earningsBody.innerHTML = '<tr class="empty-row"><td colspan="6">No earnings data. Schedule shifts to see estimates.</td></tr>';
            elements.semesterGross.textContent = '$0.00';
            elements.semesterNet.textContent = '$0.00';
            return;
        }

        const nonEmptyPeriods = earningsData.periods.filter(p => !p.isEmpty);

        if (nonEmptyPeriods.length === 0) {
            elements.earningsBody.innerHTML = '<tr class="empty-row"><td colspan="6">No scheduled shifts in pay periods.</td></tr>';
            elements.semesterGross.textContent = '$0.00';
            elements.semesterNet.textContent = '$0.00';
            return;
        }

        let html = '';
        nonEmptyPeriods.forEach(period => {
            html += '<tr>';
            html += '<td>' + period.periodLabel + '</td>';
            html += '<td>' + period.hours + 'h</td>';
            html += '<td class="positive">$' + period.gross.toFixed(2) + '</td>';
            html += '<td>$' + period.totalTax.toFixed(2) + '</td>';
            html += '<td class="positive">$' + period.net.toFixed(2) + '</td>';
            html += '<td>' + period.paycheckDate + '</td>';
            html += '</tr>';
        });
        elements.earningsBody.innerHTML = html;
        elements.semesterGross.textContent = '$' + earningsData.totals.gross.toFixed(2);
        elements.semesterNet.textContent = '$' + earningsData.totals.net.toFixed(2);
    }

    function renderClassList() {
        if (!userCalendar || !userCalendar.classes || userCalendar.classes.length === 0) {
            elements.classList.innerHTML = '<p class="empty-message">No classes added yet.</p>';
            return;
        }

        const byDay = {};
        userCalendar.classes.forEach((c, idx) => {
            if (!byDay[c.day]) byDay[c.day] = [];
            byDay[c.day].push({ ...c, index: idx });
        });

        let html = '';
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].forEach(day => {
            if (byDay[day]) {
                byDay[day].forEach(c => {
                    html += '<div class="class-item">';
                    html += '<span class="class-name">' + c.name + '</span>';
                    html += '<span class="class-time">' + day + ' ' + formatTime(c.startTime) + '-' + formatTime(c.endTime) + '</span>';
                    html += '<button class="btn-remove" data-index="' + c.index + '">&times;</button>';
                    html += '</div>';
                });
            }
        });

        elements.classList.innerHTML = html;
        elements.classList.querySelectorAll('.btn-remove').forEach(btn => {
            btn.addEventListener('click', e => removeClass(parseInt(e.target.dataset.index)));
        });
    }

    function updateSettingsInfo(stats) {
        elements.lastScraped.textContent = stats.lastScraped ? new Date(stats.lastScraped).toLocaleString() : 'Never';
        elements.daysLoaded.textContent = stats.daysScraped;
    }

    function filterShifts() {
        const dayFilter = elements.dayFilter.value;
        const statusFilter = elements.statusFilter.value;
        const rows = elements.shiftsBody.querySelectorAll('tr:not(.empty-row)');

        rows.forEach(row => {
            const rowDay = row.dataset.day;
            const rowStatus = row.dataset.status;
            const dayMatch = !dayFilter || rowDay === dayFilter;
            const statusMatch = !statusFilter || rowStatus === statusFilter;
            row.style.display = (dayMatch && statusMatch) ? '' : 'none';
        });
    }

    async function clearData() {
        if (!confirm('Clear all shift data? Class schedule will be preserved.')) return;
        try {
            await chrome.storage.local.remove(['shiftData', 'lastScraped']);
            shiftData = {};
            optimizationResult = null;
            await loadData();
            showToast('Shift data cleared');
        } catch (error) {
            showToast('Failed to clear data');
        }
    }

    function exportData() {
        const exportObj = { shiftData, userCalendar, userConfig, exportedAt: new Date().toISOString() };
        const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'shift-data-' + new Date().toISOString().split('T')[0] + '.json';
        a.click();
        URL.revokeObjectURL(url);
        showToast('Data exported');
    }

    function sendMessage(message) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage(message, response => {
                if (chrome.runtime.lastError) {
                    reject(chrome.runtime.lastError);
                } else {
                    resolve(response);
                }
            });
        });
    }

    function formatDate(dateStr) {
        if (!dateStr) return '';
        const parts = dateStr.split('-');
        return parts[1] + '/' + parts[2];
    }

    function formatTime(timeStr) {
        if (!timeStr) return '';
        const [h, m] = timeStr.split(':').map(Number);
        const ampm = h >= 12 ? 'PM' : 'AM';
        const hour = h % 12 || 12;
        return hour + ':' + String(m).padStart(2, '0') + ' ' + ampm;
    }

    function formatStatus(status) {
        const map = {
            available: 'Available',
            dropped: 'Dropped',
            taken: 'Taken',
            your_shift: 'Your Shift'
        };
        return map[status] || status;
    }

    function updateStatus(text) {
        if (elements.statusBadge) {
            elements.statusBadge.textContent = text;
        }
    }

    function showToast(message) {
        elements.toastMessage.textContent = message;
        elements.toast.classList.add('show');
        setTimeout(() => elements.toast.classList.remove('show'), 3000);
    }

    function startBulkScrape() {
        const startDate = elements.bulkStartDate.value;
        const endDate = elements.bulkEndDate.value;

        if (!startDate || !endDate) {
            showToast('Please select both start and end dates.');
            return;
        }

        if (startDate > endDate) {
            showToast('Start date must be before end date.');
            return;
        }

        elements.startBulkScrapeBtn.disabled = true;
        elements.startBulkScrapeBtn.textContent = 'Scraping...';
        elements.bulkScrapeProgress.style.display = 'block';
        updateBulkProgress({ current: 0, total: 1, date: 'Initializing...' });

        chrome.runtime.sendMessage({
            action: 'START_BULK_SCRAPE',
            data: { startDate, endDate }
        });
    }

    function updateBulkProgress(data) {
        const { current, total, date } = data;
        const percent = Math.round((current / total) * 100);

        elements.bulkScrapeStatus.textContent = `Scraping: ${date}`;
        elements.bulkScrapePercent.textContent = `${percent}%`;
        elements.bulkScrapeFill.style.width = `${percent}%`;
    }

    function finishBulkScrape(data) {
        elements.startBulkScrapeBtn.disabled = false;
        elements.startBulkScrapeBtn.textContent = 'Start Auto-Scrape';
        elements.bulkScrapeStatus.textContent = 'Complete!';
        elements.bulkScrapePercent.textContent = '100%';
        elements.bulkScrapeFill.style.width = '100%';

        showToast(`Scraped ${data.count} days successfully!`);
        loadData();
    }

    document.addEventListener('DOMContentLoaded', init);
})();
