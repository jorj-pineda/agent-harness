import assert from 'node:assert/strict';
globalThis.window={};
globalThis.sessionStorage={setItem(){throw Error('blocked');
  }};
const m=await import('./client/src/lib/shareInvite.ts');
assert.equal(m.storePendingShareInvite('a'.repeat(43)),false);
