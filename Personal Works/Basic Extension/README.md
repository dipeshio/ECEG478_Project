# Instagram Follow Analyzer Chrome Extension

A Chrome extension that helps you analyze the difference between who you follow and who follows you back on Instagram.

## Features

- 📥 **Read Followers**: Extract all your Instagram followers
- 📤 **Read Following**: Extract all accounts you're following  
- 🔍 **Show Difference**: Display who you follow but doesn't follow you back
- 💾 **Data Persistence**: Saves your data locally for quick access
- 🎨 **Modern UI**: Beautiful, responsive design with smooth animations

## Installation

1. **Download the Extension**
   - Download all files to a folder on your computer

2. **Load in Chrome**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top right)
   - Click "Load unpacked"
   - Select the folder containing the extension files

3. **Pin the Extension**
   - Click the puzzle piece icon in Chrome toolbar
   - Find "Instagram Follow Analyzer" and click the pin icon

## How to Use

### Step 1: Navigate to Instagram
- Go to [Instagram.com](https://instagram.com) and log in
- Navigate to your profile page

### Step 2: Read Your Followers
- Click on your "followers" count to open the followers list
- Click the extension icon and press "📥 Read Followers"
- The extension will automatically scroll and collect all follower usernames
- Wait for the process to complete (this may take a few minutes for large lists)

### Step 3: Read Your Following
- Click on your "following" count to open the following list  
- Click "📤 Read Following" in the extension
- Wait for the process to complete

### Step 4: Analyze the Difference
- Click "🔍 Show Who Doesn't Follow Back"
- View the list of accounts you follow but who don't follow you back

## Technical Details

### How It Works
The extension uses a content script to:
1. Find all username elements with the specific Instagram CSS classes: `_ap3a _aaco _aacw _aacx _aad7 _aade`
2. Extract the username text from these elements
3. Automatically scroll through the entire list to load all users
4. Compare the two lists to find differences

### Data Storage
- All data is stored locally in Chrome's storage
- No data is sent to external servers
- Data persists between browser sessions

### Permissions
- `activeTab`: To interact with the current Instagram tab
- `storage`: To save follower/following data locally
- `host_permissions`: To run on Instagram.com

## Troubleshooting

### Extension Not Working
- Make sure you're on Instagram.com
- Refresh the Instagram page and try again
- Check that the extension is enabled in Chrome settings

### No Users Found
- Ensure you're on the correct followers/following page
- Make sure the list is fully loaded before clicking the button
- Try scrolling manually first, then click the extension button

### Slow Performance
- Large follower lists (1000+) may take several minutes to process
- The extension shows progress updates as it works
- Don't close the tab while the extension is running

## Privacy & Security
- This extension only works locally on your device
- No personal data is transmitted to external servers
- All follower/following data stays in your browser
- The extension only reads publicly visible information from Instagram

## Support
If you encounter any issues or have suggestions for improvements, please check that:
1. You're using the latest version of Chrome
2. Instagram hasn't changed their page structure
3. You have a stable internet connection during the extraction process

## Version History
- **v1.0**: Initial release with basic follower analysis functionality
