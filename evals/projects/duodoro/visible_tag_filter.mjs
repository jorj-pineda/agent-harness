import assert from 'node:assert/strict';
const m=await import('./client/src/lib/focusTags.ts');
assert.equal(m.normalizeTagFilter('work'),'work');
assert.equal(m.normalizeTagFilter(null),'all');
