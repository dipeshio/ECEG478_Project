class FollowAnalyzer {
  constructor() {
    this.followers = [];
    this.following = [];
    this.initializeEventListeners();
    this.loadStoredData();
  }

  initializeEventListeners() {
    document.getElementById('readFollowers').addEventListener('click', () => {
      this.readFollowers();
    });

    document.getElementById('readFollowing').addEventListener('click', () => {
      this.readFollowing();
    });

    document.getElementById('showDifference').addEventListener('click', () => {
      this.showDifference();
    });
  }

  async loadStoredData() {
    try {
      const result = await chrome.storage.local.get(['followers', 'following']);
      if (result.followers) {
        this.followers = result.followers;
        this.updateStatus(`Loaded ${this.followers.length} followers from storage`, 'success');
      }
      if (result.following) {
        this.following = result.following;
        this.updateStatus(`Loaded ${this.following.length} following from storage`, 'success');
      }
    } catch (error) {
      console.error('Error loading stored data:', error);
    }
  }

  async saveData(type, data) {
    try {
      await chrome.storage.local.set({ [type]: data });
    } catch (error) {
      console.error('Error saving data:', error);
    }
  }

  updateStatus(message, type = 'loading') {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
  }

  async getCurrentTab() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    return tab;
  }

  async checkInstagramPage() {
    const tab = await this.getCurrentTab();
    if (!tab.url.includes('instagram.com')) {
      throw new Error('Please navigate to Instagram first');
    }
    return tab;
  }

  async readFollowers() {
    try {
      this.updateStatus('Reading followers... Please wait while we scroll through your followers list.', 'loading');
      this.disableButtons();

      const tab = await this.checkInstagramPage();
      
      // Send message to content script
      const response = await chrome.tabs.sendMessage(tab.id, {
        action: 'extractFollowers'
      });

      if (response.success) {
        this.followers = response.usernames;
        await this.saveData('followers', this.followers);
        this.updateStatus(`Successfully read ${this.followers.length} followers!`, 'success');
        this.displayResults('Followers', this.followers);
      } else {
        throw new Error(response.error || 'Failed to extract followers');
      }
    } catch (error) {
      this.updateStatus(`Error reading followers: ${error.message}`, 'error');
      console.error('Error:', error);
    } finally {
      this.enableButtons();
    }
  }

  async readFollowing() {
    try {
      this.updateStatus('Reading following... Please wait while we scroll through your following list.', 'loading');
      this.disableButtons();

      const tab = await this.checkInstagramPage();
      
      // Send message to content script
      const response = await chrome.tabs.sendMessage(tab.id, {
        action: 'extractFollowing'
      });

      if (response.success) {
        this.following = response.usernames;
        await this.saveData('following', this.following);
        this.updateStatus(`Successfully read ${this.following.length} following!`, 'success');
        this.displayResults('Following', this.following);
      } else {
        throw new Error(response.error || 'Failed to extract following');
      }
    } catch (error) {
      this.updateStatus(`Error reading following: ${error.message}`, 'error');
      console.error('Error:', error);
    } finally {
      this.enableButtons();
    }
  }

  showDifference() {
    if (this.followers.length === 0 || this.following.length === 0) {
      this.updateStatus('Please read both followers and following lists first!', 'error');
      return;
    }

    // Find people you follow but who don't follow you back
    const followersSet = new Set(this.followers);
    const notFollowingBack = this.following.filter(user => !followersSet.has(user));

    this.updateStatus(`Found ${notFollowingBack.length} people who don't follow you back`, 'success');
    this.displayResults('Not Following Back', notFollowingBack, true);
  }

  displayResults(title, usernames, isDifference = false) {
    const resultsDiv = document.getElementById('results');
    
    if (usernames.length === 0) {
      resultsDiv.innerHTML = `
        <div class="count">${title}: No users found</div>
      `;
      return;
    }

    const usernameList = usernames.map(username => 
      `<div class="username${isDifference ? ' difference' : ''}">${username}</div>`
    ).join('');

    resultsDiv.innerHTML = `
      <div class="count">${title} (${usernames.length})</div>
      <div class="username-list">${usernameList}</div>
    `;
  }

  disableButtons() {
    document.querySelectorAll('button').forEach(btn => btn.disabled = true);
  }

  enableButtons() {
    document.querySelectorAll('button').forEach(btn => btn.disabled = false);
  }
}

// Listen for progress updates from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'progress') {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<div class="status loading">Reading ${request.type}... Found ${request.count} users so far</div>`;
  }
});

// Initialize when popup loads
document.addEventListener('DOMContentLoaded', () => {
  new FollowAnalyzer();
});
