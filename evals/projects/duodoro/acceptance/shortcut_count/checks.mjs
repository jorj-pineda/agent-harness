import assert from 'node:assert/strict';
const {startShortcutAction:f}=await import('./client/src/lib/keyboardShortcuts.ts');
const base={inGame:true,sessionId:'room',connected:true,playerCount:1,sessionStarted:false,intentionEditing:false,phase:'waiting'};
for(const count of [NaN,Infinity,-Infinity,-1,0,0.5,1.5,Number.MAX_SAFE_INTEGER+1,'1',true,false,null,undefined,[],{}]) assert.equal(f(Object.freeze({...base,playerCount:count})),null);
for(const count of [1,2,3,Number.MAX_SAFE_INTEGER]) {
  assert.equal(f(Object.freeze({...base,playerCount:count})),'start');
  assert.equal(f({...base,phase:'ready',playerCount:count}),'again');
}
for(const field of ['inGame','connected'])assert.equal(f({...base,[field]:false}),null);
for(const field of ['sessionStarted','intentionEditing'])assert.equal(f({...base,[field]:true}),null);
for(const id of [null,''])assert.equal(f({...base,sessionId:id}),null);
for(const phase of ['focus','break','ended','unknown'])assert.equal(f({...base,phase}),null);
