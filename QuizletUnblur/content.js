// content.js
(function() {
    // Function to apply the necessary changes
    function unblurQuizlet() {
      // Target the markdown-body div elements
      const markdownBodies = document.querySelectorAll('div.markdown-body');
      
      if (markdownBodies.length > 0) {
        console.log('Found markdown-body elements, removing blur...');
        
        // For each markdown-body element
        markdownBodies.forEach(body => {
          // Apply CSS to remove blur effects
          const style = document.createElement('style');
          style.textContent = `
            div.markdown-body * {
              filter: none !important;
              -webkit-user-select: auto !important;
              user-select: auto !important;
            }
            
            .b1xkd811 {
              filter: none !important;
              -webkit-user-select: auto !important;
              user-select: auto !important;
            }
          `;
          document.head.appendChild(style);
        });
      }
      
      // Remove elements with class o3dpi86 and pxrylku
      const elementsToRemove = document.querySelectorAll('.o3dpi86, .pxrylku');
      if (elementsToRemove.length > 0) {
        console.log(`Removing ${elementsToRemove.length} elements with specified classes...`);
        elementsToRemove.forEach(el => {
          el.remove();
        });
      }
    }
    
    // Run the function initially
    unblurQuizlet();
    
    // Set up a mutation observer to handle dynamically loaded content
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
          unblurQuizlet();
        }
      }
    });
    
    // Start observing the document with the configured parameters
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Also run after a short delay to catch any elements that might load late
    setTimeout(unblurQuizlet, 1500);
  })();