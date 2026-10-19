import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const client=await import('./client/src/lib/rotation.ts');
const server=require(process.cwd()+'/server/rotation.js');
const golden=JSON.parse(readFileSync(process.env.DUODORO_GOLDEN,'utf8'));
for(const m of [client,server]) {
  assert.equal(typeof m.upcomingWorlds,'function');
  assert.deepEqual(m.ROTATION_WORLDS,golden.worlds);
  assert.equal(m.CYCLE_LENGTH,8);
  assert.equal(m.ROTATION_MS,3600000);
  assert.equal(m.ROTATION_OFFSET_MS,1800000);
  for(const row of golden.cycles)assert.deepEqual(m.orderForCycle(row.cycle),row.order);
  for(const row of golden.moments) {
    assert.equal(m.slotAt(row.now),row.slot);
    assert.equal(m.worldAt(row.now),row.world);
    assert.equal(m.nextRotationAt(row.now),row.next);
    assert.equal(m.msUntilRotation(row.now),row.remaining);
    for(const count of [0,1,2,3,7,8])assert.deepEqual(m.upcomingWorlds(row.now,count),row.preview.slice(0,count));
    assert.deepEqual(m.upcomingWorlds(row.now),row.preview.slice(0,3));
    const list=m.upcomingWorlds(row.now,2);
    list[0].world='mutated';
    list.pop();
    assert.deepEqual(m.upcomingWorlds(row.now,2),row.preview.slice(0,2));
  }
  for(const now of [NaN,Infinity,-Infinity,null,undefined,'0',true,{},[]])assert.throws(()=>m.upcomingWorlds(now),RangeError);
  for(const count of [-1,0.5,9,NaN,Infinity,null,'2',true,{},[]])assert.throws(()=>m.upcomingWorlds(0,count),RangeError);
  assert.deepEqual(m.ROTATION_WORLDS,golden.worlds);
}
for(const row of golden.moments)assert.deepEqual(client.upcomingWorlds(row.now,8),server.upcomingWorlds(row.now,8));
