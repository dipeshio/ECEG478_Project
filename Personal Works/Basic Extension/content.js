// Content script to extract usernames from Instagram
class InstagramUserExtractor {
  constructor() {
    this.usernames = new Set();
  }

  // Extract usernames from the current view
  extractUsernames() {
    // Look for spans with the specific class pattern
    const usernameElements = document.querySelectorAll('span._ap3a._aaco._aacw._aacx._aad7._aade[dir="auto"]');
    const usernames = [];
    
    usernameElements.forEach(element => {
      const username = element.textContent.trim();
      if (username && !usernames.includes(username)) {
        usernames.push(username);
      }
    });
    
    return usernames;
  }

  // Scroll and extract usernames progressively
  async extractAllUsernames(type) {
    const allUsernames = new Set();
    let previousCount = 0;
    let stableCount = 0;
    const maxStableIterations = 3;
    
    // Function to scroll to bottom
    const scrollToBottom = () => {
      const scrollableContainer = document.querySelector('div[style*="overflow"]') || 
                                 document.querySelector('.x9f619.xjbqb8w.x78zum5.x168nmei.x13lgxp2.x5pf9jr.xo71vjh.x1n2onr6.x1plvlek.xryxfnj.x1c4vz4f.x2lah0s.xdt5ytf.xqjyukv.x1qjc9v5.x1oa3qoh.x1nhvcw1') ||
                                 document.body;
      
      scrollableContainer.scrollTop = scrollableContainer.scrollHeight;
    };

    while (stableCount < maxStableIterations) {
      // Extract current usernames
      const currentUsernames = this.extractUsernames();
      currentUsernames.forEach(username => allUsernames.add(username));
      
      // Check if we're getting new usernames
      if (allUsernames.size === previousCount) {
        stableCount++;
      } else {
        stableCount = 0;
        previousCount = allUsernames.size;
      }
      
      // Scroll down to load more
      scrollToBottom();
      
      // Wait for content to load
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Send progress update
      chrome.runtime.sendMessage({
        action: 'progress',
        type: type,
        count: allUsernames.size
      });
    }
    
    return Array.from(allUsernames);
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const extractor = new InstagramUserExtractor();
  
  if (request.action === 'extractFollowers') {
    extractor.extractAllUsernames('followers').then(usernames => {
      sendResponse({ success: true, usernames: usernames });
    }).catch(error => {
      sendResponse({ success: false, error: error.message });
    });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'extractFollowing') {
    extractor.extractAllUsernames('following').then(usernames => {
      sendResponse({ success: true, usernames: usernames });
    }).catch(error => {
      sendResponse({ success: false, error: error.message });
    });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'quickExtract') {
    const usernames = extractor.extractUsernames();
    sendResponse({ success: true, usernames: usernames });
  }
});

// Optional: Add visual indicator when extension is active
const indicator = document.createElement('div');
indicator.style.cssText = `
  position: fixed;
  top: 10px;
  right: 10px;
  background: #4CAF50;
  color: white;
  padding: 8px 12px;
  border-radius: 20px;
  font-size: 12px;
  z-index: 10000;
  font-family: Arial, sans-serif;
  display: none;
`;
indicator.textContent = '📊 Follow Analyzer Active';
document.body.appendChild(indicator);

// Show indicator briefly when content script loads
indicator.style.display = 'block';
setTimeout(() => {
  indicator.style.display = 'none';
}, 3000);
