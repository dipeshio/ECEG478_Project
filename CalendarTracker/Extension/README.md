# Shift Scheduler Optimizer

A Chrome Extension (Manifest V3) for scraping shift data from Bucknell's Conportal circulation schedule and recommending optimal work schedules based on student class constraints.

## Features

- **Automatic Shift Scraping**: Extracts shift data directly from the Conportal page
- **Smart Optimization**: Recommends shifts based on:
  - Target hours (13-15h/week)
  - Preferred time (4-10 PM)
  - Class schedule conflicts
- **Excel-like Interface**: View all shifts in a clean spreadsheet format
- **Local Storage**: All data stored locally using `chrome.storage.local`

## Installation

1. **Download/Clone** this extension folder to your computer

2. **Open Chrome Extensions**:
   - Navigate to `chrome://extensions/` in Chrome
   - Or: Menu → More Tools → Extensions

3. **Enable Developer Mode**:
   - Toggle the "Developer mode" switch in the top-right corner

4. **Load the Extension**:
   - Click "Load unpacked"
   - Select the `Extension` folder containing `manifest.json`

5. **Pin the Extension** (optional):
   - Click the puzzle piece icon in Chrome toolbar
   - Pin "Shift Scheduler Optimizer" for easy access

## Usage

1. **Scrape Shifts**:
   - Visit: `https://www.linux.bucknell.edu/~conportal/circ/show_shifts.php`
   - The extension automatically scrapes data when you visit
   - Navigate to different days to collect more shift data

2. **View & Analyze**:
   - Click the extension icon in Chrome toolbar
   - View all scraped shifts in the "All Shifts" tab
   - Click "Analyze Schedule" to get recommendations

3. **Recommended Shifts**:
   - Gold-highlighted rows = Optimal shifts
   - Score indicates how well a shift fits your schedule
   - Positive scores = good fits, Negative = avoid

## Scoring System

| Condition | Score |
|-----------|-------|
| Preferred time (4-10 PM) | +10 |
| Helps hit 13-15h target | +5 |
| Dropped shift (easier to get) | +3 |
| Before 4 PM (when not needed) | -50 |
| Conflicts with class | -100 |

## Files

```
Extension/
├── manifest.json      # Extension configuration
├── content.js         # DOM scraping logic
├── background.js      # Service worker & data management
├── optimizer.js       # Schedule optimization algorithm
├── popup.html         # UI structure
├── popup.css          # Spreadsheet styling
├── popup.js           # UI logic
├── icons/             # Extension icons
└── README.md          # This file
```

## Notes

- **January 19th (MLK Day)**: Shifts not available for selection on this date
- **School Year Start**: January 20th, 2026 (Tuesday)
- Data is stored locally only - no external servers used

## Troubleshooting

**No data showing?**
- Make sure you've visited the Conportal shift schedule page
- Check that the extension is enabled

**Wrong class schedule?**
- The default schedule is hardcoded in `background.js`
- Edit the `getDefaultCalendar()` function to update your classes

## License

For personal/educational use only.
