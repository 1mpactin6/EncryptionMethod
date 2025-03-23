/**
 * This script types for you automatically on www.typingclub.com:
 * 1. Open the website
 * 2. Blaze past the tutorials
 * 3. Go into a level
 * 4. Open Console
 * 5. Paste the script and press ENTER
 */

// NOTE: When delay (in ms between two strokes) is too low, the site might bug out and the result page will not be shown
const minDelay = 60;
const maxDelay = 60;
const accuracy = 0.95; // Set the desired accuracy (e.g., 95% correct)

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

function recordKey(chr) {
  // send it straight to the internal API
  window.core.record_keydown_time(chr);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function introduceErrors(chrs) {
  const totalErrors = Math.floor(chrs.length * (1 - accuracy));
  for (let i = 0; i < totalErrors; i++) {
    const index = Math.floor(Math.random() * chrs.length);
    const originalChar = chrs[index];
    chrs[index] = String.fromCharCode(Math.floor(Math.random() * 26) + 97); // random lowercase letter
    chrs.splice(index + 1, 0, originalChar); // add original character after the error
  }
  return chrs;
}

async function autoPlay(finish) {
  let chrs = getTargetCharacters();
  chrs = introduceErrors(chrs);
  for (let i = 0; i < chrs.length - (!finish); ++i) {
    const c = chrs[i];
    recordKey(c);
    await sleep(Math.random() * (maxDelay - minDelay) + minDelay);
  }
}

// ############################################################################################################
// go!
// ############################################################################################################

autoPlay(true);