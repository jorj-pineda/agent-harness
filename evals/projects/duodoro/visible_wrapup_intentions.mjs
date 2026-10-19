import assert from 'node:assert/strict';
const m=await import('./client/src/lib/sessionWrapUp.ts');
assert.deepEqual(m.unfinishedIntentions('u','focus',{current:{u:{text:' work ',completed:false}},next:{u:'work'}},null),['work']);
