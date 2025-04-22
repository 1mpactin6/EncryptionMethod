/**
 * This script types for you automatically on www.typingclub.com:
 * 1. Open the website
 * 2. Go into a level
 * 3. Open Console (F12)
 * 4. Paste the script and press ENTER
 *
 * NOTE: This version simplifies error simulation to potentially reduce detection issues.
 */

// --- Configuration ---

// Minimum delay in milliseconds before starting the typing after the script is pasted
const initialStartDelay = 1500; // Increased initial delay slightly

// Minimum delay in milliseconds between typing strokes (from keyup to next keydown)
let minDelay = 45; // Adjusted min delay
// Maximum delay in milliseconds between typing strokes
let maxDelay = 50; // Adjusted maxDelay

// Minimum delay in milliseconds between keydown, keypress, and keyup events for DOM simulation
const minKeyEventDelay = 10; // Added small delay between key events
// Maximum delay in milliseconds between keydown, keypress, and keyup events for DOM simulation
const maxKeyEventDelay = 30; // Added small delay between key events

// Target accuracy percentage (0 to 100). This is the overall chance of *an* error occurring for a character.
const accuracyPercentage = 98; // Set your desired accuracy here

// --- Logging Configuration ---

// How often to log progress to the console (in characters processed from the lesson)
const progressLogInterval = 15;

// Enable detailed logging for debugging (true/false)
const enableDetailedLogging = false;

// --- Global Control ---

// Set this variable to true in the console to stop the script manually:
// window.stopAutoTyper = true;
window.stopAutoTyper = false;

// --- User Input Detection for Stopping ---

// Function to handle user key presses
function handleUserInputStop(event) {
    // Stop the script if the user presses any key.
    console.log(`[AutoTyper] User key pressed (${event.key}). Stopping.`);
    window.stopAutoTyper = true;
    // Optionally remove the event listener after the first key press
    window.removeEventListener('keydown', handleUserInputStop);
}

// Add event listener to detect user key presses
window.addEventListener('keydown', handleUserInputStop);


// --- Character Mapping and Helpers ---

const keyOverrides = {
  [String.fromCharCode(160)]: ' '    // convert hardspace to normal space
};

function getTargetCharacters() {
  const els = Array.from(document.querySelectorAll('.token span.token_unit'));
  const chrs = els
    .map(el => {
      // get letter to type from each letter DOM element
      if (el.firstChild?.classList?.contains('_enter')) {
        // special case: ENTER
        return '\n';
      }
      let text = el.textContent[0];
      return text;
    })
    .map(c => keyOverrides.hasOwnProperty(c) ? keyOverrides[c] : c); // convert special characters
  return chrs;
}

// Function to get the currently active character element in the DOM
function getActiveCharElement() {
    return document.querySelector('.token span.token_unit.token_unit--active');
}

// Function to get the index of a character element in the DOM
function getCharIndexInDOM(element) {
    const els = Array.from(document.querySelectorAll('.token span.token_unit'));
    return els.indexOf(element);
}

// Flag to indicate if the internal API is available
let useInternalApi = false;

/**
 * Records a key press using either the internal API or revised DOM simulation.
 * Handles regular characters. Backspace simulation is removed in this version.
 * @param {string} chr The character to type.
 */
async function recordKey(chr) {
    if (useInternalApi) {
        if (enableDetailedLogging) console.log(`[AutoTyper] Using internal API for "${chr}"`);
        // TypingClub's internal API might handle backspace directly if passed, but we are simplifying errors
        window.core.record_keydown_time(chr);
    } else {
        if (enableDetailedLogging) console.log(`[AutoTyper] Using DOM simulation for "${chr}"`);
        const activeElement = document.activeElement;
        const targetElement = (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) ? activeElement : document.body;
        // Always simulate regular character key events in DOM fallback
        simulateDOMKey(chr, targetElement);
    }
}

/**
 * Simulates keydown, keypress, and keyup DOM events consecutively for a regular character.
 * @param {string} chr The character for the event.
 * @param {Element} el The target DOM element.
 */
function simulateDOMKey(chr, el) {
    try {
        // Simulate keydown
        const keyDownEvent = new KeyboardEvent('keydown', {
            key: chr,
            code: `Key${chr.toUpperCase()}`, // This might need refinement based on the actual key
            charCode: chr.charCodeAt(0),
            keyCode: chr.charCodeAt(0),
            which: chr.charCodeAt(0),
            bubbles: true,
            cancelable: true,
            isTrusted: true // Attempt to simulate trusted event
        });
        el.dispatchEvent(keyDownEvent);

        // Simulate keypress
        const keyPressEvent = new KeyboardEvent('keypress', {
            key: chr,
            code: `Key${chr.toUpperCase()}`,
            charCode: chr.charCodeAt(0),
            keyCode: chr.charCodeAt(0),
            which: chr.charCodeAt(0),
            bubbles: true,
            cancelable: true,
             isTrusted: true
        });
         el.dispatchEvent(keyPressEvent);

         // Simulate keyup
         const keyUpEvent = new KeyboardEvent('keyup', {
            key: chr,
            code: `Key${chr.toUpperCase()}`,
            charCode: chr.charCodeAt(0),
            keyCode: chr.charCodeAt(0),
            which: chr.charCodeAt(0),
            bubbles: true,
            cancelable: true,
             isTrusted: true
        });
         el.dispatchEvent(keyUpEvent);

    } catch (e) {
        console.error("[AutoTyper] Failed to simulate DOM key event:", e);
    }
}


function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// Generates a random printable ASCII character (excluding space and delete-like chars)
function getRandomIncorrectChar() {
    const minAscii = 33; // '!'
    const maxAscii = 126; // '~'
    let charCode;
    do {
        charCode = Math.floor(Math.random() * (maxAscii - minAscii + 1)) + minAscii;
    } while (String.fromCharCode(charCode) === ' '); // Avoid space as an incorrect char if possible
    return String.fromCharCode(charCode);
}


// --- Auto Play Function ---

async function autoPlay() {
  console.log(`[AutoTyper] Script loaded. Waiting for ${initialStartDelay}ms before starting...`);
  await sleep(initialStartDelay);

  // Check if internal API is available
  if (window.core && window.core.record_keydown_time) {
      useInternalApi = true;
      console.log("[AutoTyper] TypingClub internal API detected. Using internal method for typing.");
  } else {
      useInternalApi = false;
      console.warn("[AutoTyper] TypingClub internal API not found. Falling back to DOM simulation. Error simulation will be simplified.");
  }


  const chrs = getTargetCharacters();

  if (!chrs || chrs.length === 0) {
      console.error("[AutoTyper] Could not find target characters. Make sure you are in a lesson and the page is fully loaded.");
      // Remove the user input listener if the script can't start
      window.removeEventListener('keydown', handleUserInputStop);
      return; // Stop execution if no characters are found
  }

  const totalCharacters = chrs.length;
  let intendedCharactersProcessed = 0; // Counts characters from the lesson that the script has processed
  let intentionalErrorsIntroduced = 0; // Counts instances where an incorrect character was intentionally typed
  const startTime = Date.now();

  console.log(`[AutoTyper] Starting to type ${totalCharacters} characters...`);
  console.log(`[AutoTyper] Target Accuracy: ${accuracyPercentage}%`);
  console.log(`[AutoTyper] Speed Range (between keyups): ${minDelay}-${maxDelay}ms`);
  console.log(`[AutoTyper] Error Simulation: Typing a single incorrect character when an error is intended.`);
  console.log(`[AutoTyper] Script will stop if you press any key.`);
  console.log(`[AutoTyper] To stop manually via console, type 'window.stopAutoTyper = true;'`);


  for (let i = 0; i < totalCharacters; ++i) {
    // Check for stop (manual or user input)
    if (window.stopAutoTyper) {
        console.log("[AutoTyper] Stop requested. Stopping.");
        break;
    }

    const correctChar = chrs[i];
    intendedCharactersProcessed++; // Increment for each character processed from the lesson

    // Determine if an error should occur
    const shouldMakeError = Math.random() * 100 > accuracyPercentage;

    if (shouldMakeError) {
      intentionalErrorsIntroduced++; // Count this as an intentional error occurrence
      const incorrectChar = getRandomIncorrectChar();
      if (enableDetailedLogging) console.log(`[AutoTyper] Intentionally typing incorrect character: "${incorrectChar}" instead of "${correctChar}"`);

      // Always type a single incorrect character when an error is intended
      await recordKey(incorrectChar);

    } else {
      // Type the correct character
      await recordKey(correctChar);
    }

    // Log progress periodically
    if (intendedCharactersProcessed % progressLogInterval === 0 || i === totalCharacters - 1) { // Log on last character too
        const currentTime = Date.now();
        const elapsedTimeInSeconds = (currentTime - startTime) / 1000;
        // Ensure elapsedTimeInSeconds is not zero to avoid division by zero
        const currentSpeedWPM = elapsedTimeInSeconds > 0 ? (intendedCharactersProcessed / 5) / (elapsedTimeInSeconds / 60) : 0; // Rough WPM calculation based on intended characters
        // Accuracy calculation: (Total intended characters processed - Intentional errors introduced) / Total intended characters processed
        // Note: This accuracy calculation now reflects the percentage of characters where an *intentional* error was NOT introduced.
        const currentAccuracy = intendedCharactersProcessed > 0 ? ((intendedCharactersProcessed - intentionalErrorsIntroduced) / intendedCharactersProcessed) * 100 : 100;


        console.log(`[AutoTyper] Progress: ${intendedCharactersProcessed}/${totalCharacters} characters processed | Intentional Errors: ${intentionalErrorsIntroduced} | Accuracy: ${currentAccuracy.toFixed(2)}% | Speed: ${currentSpeedWPM.toFixed(2)} WPM`);
    }


    // Wait for a random delay before the next character (from keyup of the last typed char to keydown of the next)
    if (i < totalCharacters - 1) {
        await sleep(Math.random() * (maxDelay - minDelay) + minDelay);
    }
  }

  const endTime = Date.now();
  const totalElapsedTimeInSeconds = (endTime - startTime) / 1000;
   // Ensure totalElapsedTimeInSeconds is not zero to avoid division by zero
  const finalSpeedWPM = totalElapsedTimeInSeconds > 0 ? (intendedCharactersProcessed / 5) / (totalElapsedTimeInSeconds / 60) : 0;
  const finalAccuracy = intendedCharactersProcessed > 0 ? ((intendedCharactersProcessed - intentionalErrorsIntroduced) / intendedCharactersProcessed) * 100 : 100;


  console.log(`[AutoTyper] Auto-typing finished.`);
  console.log(`[AutoTyper] Final Stats: Total Characters Processed: ${intendedCharactersProcessed} | Intentional Errors: ${intentionalErrorsIntroduced} | Final Accuracy: ${finalAccuracy.toFixed(2)}% | Final Speed: ${finalSpeedWPM.toFixed(2)} WPM`);

  // Remove the user input listener when the script finishes
  window.removeEventListener('keydown', handleUserInputStop);

}

// ############################################################################################################
// go!
// ############################################################################################################

// Start the autoplay process
autoPlay();
