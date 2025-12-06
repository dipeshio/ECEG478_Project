let currentNotFollowingBack = [];

// Load saved data on popup open
document.addEventListener('DOMContentLoaded', async () => {
  await updateStats();
});

document.getElementById('extractFollowers').addEventListener('click', async () => {
  await extractAndSave('followers', 'Followers');
});

document.getElementById('extractFollowing').addEventListener('click', async () => {
  await extractAndSave('following', 'Following');
});

document.getElementById('compare').addEventListener('click', async () => {
  await compareFollows();
});

document.getElementById('clear').addEventListener('click', async () => {
  await chrome.storage.local.clear();
  showStatus('All saved data cleared!', 'info');
  await updateStats();
  document.getElementById('results').classList.add('hidden');
  document.getElementById('download').classList.add('hidden');
});

document.getElementById('download').addEventListener('click', () => {
  downloadTextFile(currentNotFollowingBack, 'not_following_back');
});

async function extractAndSave(type, label) {
  const button = document.getElementById(`extract${label}`);
  const status = document.getElementById('status');
  
  button.disabled = true;
  showStatus(`Extracting ${label.toLowerCase()}...`, 'info');
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: extractUsernames
    });
    
    const usernames = results[0].result;
    
    if (usernames.length === 0) {
      showStatus(`No ${label.toLowerCase()} found on this page.`, 'error');
    } else {
      // Save to storage
      await chrome.storage.local.set({ [type]: usernames });
      showStatus(`✓ Saved ${usernames.length} ${label.toLowerCase()}!`, 'success');
      await updateStats();
    }
  } catch (error) {
    showStatus(`Error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
  }
}

async function compareFollows() {
  const status = document.getElementById('status');
  const resultsDiv = document.getElementById('results');
  const downloadBtn = document.getElementById('download');
  
  try {
    const data = await chrome.storage.local.get(['followers', 'following']);
    
    if (!data.followers || !data.following) {
      showStatus('Please extract both followers and following lists first!', 'warning');
      return;
    }
    
    const followersSet = new Set(data.followers);
    const notFollowingBack = data.following.filter(user => !followersSet.has(user));
    
    currentNotFollowingBack = notFollowingBack;
    
    if (notFollowingBack.length === 0) {
      showStatus('Great news! Everyone you follow follows you back! 🎉', 'success');
      resultsDiv.classList.add('hidden');
      downloadBtn.classList.add('hidden');
    } else {
      showStatus(`Found ${notFollowingBack.length} user(s) who don't follow you back:`, 'info');
      
      resultsDiv.innerHTML = notFollowingBack
        .map(username => `<div class="username-item">${escapeHtml(username)}</div>`)
        .join('');
      
      resultsDiv.classList.remove('hidden');
      downloadBtn.classList.remove('hidden');
    }
  } catch (error) {
    showStatus(`Error: ${error.message}`, 'error');
  }
}

async function updateStats() {
  const data = await chrome.storage.local.get(['followers', 'following']);
  const statsSection = document.getElementById('statsSection');
  
  const followersCount = data.followers ? data.followers.length : 0;
  const followingCount = data.following ? data.following.length : 0;
  
  document.getElementById('followersCount').textContent = followersCount;
  document.getElementById('followingCount').textContent = followingCount;
  
  if (followersCount > 0 || followingCount > 0) {
    statsSection.classList.remove('hidden');
  } else {
    statsSection.classList.add('hidden');
  }
}

function extractUsernames() {
  // Find all span elements with the specific class pattern
  const spans = document.querySelectorAll('span._ap3a._aaco._aacw._aacx._aad7._aade[dir="auto"]');
  
  // Extract text content from each span and remove duplicates
  const usernames = [...new Set(
    Array.from(spans)
      .map(span => span.textContent.trim())
      .filter(text => text.length > 0)
  )];
  
  return usernames;
}

function downloadTextFile(usernames, prefix) {
  const content = usernames.join('\n');
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const filename = `${prefix}_${timestamp}.txt`;
  
  chrome.downloads.download({
    url: url,
    filename: filename,
    saveAs: true
  });
}

function showStatus(message, className) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = className;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}