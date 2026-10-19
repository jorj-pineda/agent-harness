import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),m=require(process.cwd()+'/server/payloadParsers.js');
assert.equal(typeof m.boundedDuration,'function');
const values=[undefined,null,'',0,false,true,NaN,Infinity,-Infinity,-5,1,29.9,60.5,300.99,1500.5,9000,'100.75','bad'];
for(const value of values) {
  for(const [fallback,min,max] of [[1500,60,7200],[300,30,3600]]) {
    const expected=Math.min(max,Math.max(min,Number(value)||fallback));
    assert.equal(m.boundedDuration(value,fallback,min,max),expected);
  }
  for(const mode of ['flow','pomodoro',undefined,'unknown',null,5]) {
    const result=m.parseStartSession(Object.freeze({sessionId:'room',focusDuration:value,breakDuration:value,mode}));
    assert.deepEqual(result,{ok:true,value:{sessionId:'room',focusDuration:Math.min(7200,Math.max(60,Number(value)||1500)),breakDuration:Math.min(3600,Math.max(30,Number(value)||300)),mode:mode==='flow'?'flow':'pomodoro'}});
  }
}
assert.throws(()=>m.boundedDuration(Symbol('bad'),1500,60,7200),TypeError);
assert.throws(()=>m.parseStartSession({sessionId:'room',focusDuration:Symbol('bad')}),TypeError);
assert.throws(()=>m.parseStartSession({sessionId:'room',breakDuration:Symbol('bad')}),TypeError);
for(const sessionId of [undefined,null,1,{},[]])assert.deepEqual(m.parseStartSession({sessionId}),{ok:false,reason:'session_id'});
assert.equal(m.MAX_FOCUS,7200);
assert.equal(m.MAX_BREAK,3600);
const golden=JSON.parse(readFileSync(process.env.DUODORO_GOLDEN,'utf8'));
for(const row of golden)assert.deepEqual(m[row.name](...row.args),row.result);
