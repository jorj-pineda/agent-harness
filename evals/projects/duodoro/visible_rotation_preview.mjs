import assert from 'node:assert/strict';
const m=await import('./client/src/lib/rotation.ts');
assert.deepEqual(m.upcomingWorlds(0,1),[{world:m.worldAt(1800000),startsAt:1800000}]);
