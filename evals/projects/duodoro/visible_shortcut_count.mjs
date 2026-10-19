import assert from 'node:assert/strict';
const m=await import('./client/src/lib/keyboardShortcuts.ts');
assert.equal(m.startShortcutAction({inGame:true,sessionId:'s',connected:true,playerCount:NaN,sessionStarted:false,intentionEditing:false,phase:'waiting'}),null);
