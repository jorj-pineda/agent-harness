import assert from 'node:assert/strict';
globalThis.window={};
let raw=null;
globalThis.localStorage={getItem(){return raw;
  },setItem(k,v){raw=v;
  }};
const m=await import('./client/src/lib/timerPrefs.ts');
m.writeTimerPrefs({focus:53});
assert.deepEqual(JSON.parse(raw),{mode:'pomodoro',focus:55,break:5});
