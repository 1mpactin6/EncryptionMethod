const minDelay = 60;
const maxDelay = 60;
const accuracy = 0.95;

const keyOverrides = {
  [String.fromCharCode(160)]: ' '
};

function getTargetCharacters() {
  const els = Array.from(document.querySelectorAll('.token span.token_unit'));
  const chrs = els
    .map(el => {
      if (el.firstChild?.classList?.contains('_enter')) {
        return '\n';
      }
      let text = el.textContent[0];
      return text;
    })
    .map(c => keyOverrides.hasOwnProperty(c) ? keyOverrides[c] : c); // convert special characters
  return chrs;
}

function recordKey(chr) {
  window.core.record_keydown_time(chr);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function introduceErrors(chrs) {
  const totalErrors = Math.floor(chrs.length * (1 - accuracy));
  for (let i = 0; i < totalErrors; i++) {
    const index = Math.floor(Math.random() * chrs.length);
    chrs[index] = String.fromCharCode(Math.floor(Math.random() * 26) + 97); // random lowercase letter
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

autoPlay(true);