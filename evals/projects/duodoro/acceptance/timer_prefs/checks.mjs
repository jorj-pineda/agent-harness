import assert from 'node:assert/strict';
const m=await import('./client/src/lib/timerPrefs.ts');
const defaults={mode:'pomodoro',focus:25,break:5};
assert.deepEqual(m.DEFAULT_TIMER_PREFS,defaults);
assert.deepEqual([m.FOCUS_MIN,m.FOCUS_MAX,m.FOCUS_STEP,m.BREAK_MIN,m.BREAK_MAX],[5,120,5,1,30]);
delete globalThis.window;
assert.deepEqual(m.readTimerPrefs(),defaults);
assert.doesNotThrow(()=>m.writeTimerPrefs({focus:55}));
globalThis.window={};
let stored;
let calls=0;
globalThis.localStorage={getItem(k){assert.equal(k,'duodoro-timer-prefs');
    return stored;
  },setItem(k,v){assert.equal(k,'duodoro-timer-prefs');
    stored=v;
    calls++;
  }};
const previous={mode:'flow',focus:50,break:10};
const rows=[
[{focus:53},{mode:'flow',focus:55,break:10}],
[{focus:-2,break:0},{mode:'flow',focus:5,break:1}],
[{focus:999,break:99},{mode:'flow',focus:120,break:30}],
[{focus:NaN,break:Infinity},{mode:'flow',focus:25,break:5}],
[{focus:'60',break:false,mode:'bad'},{mode:'pomodoro',focus:25,break:5}],
[{focus:undefined,break:null},{mode:'flow',focus:25,break:5}],
[{mode:'pomodoro',break:2.6},{mode:'pomodoro',focus:50,break:3}],
[{focus:7.5},{mode:'flow',focus:10,break:10}],
[{},{...previous}],
];
for(const [partial,expected] of rows) {
  stored=JSON.stringify(previous);
  Object.freeze(partial);
  const oldCalls=calls;
  m.writeTimerPrefs(partial);
  assert.equal(calls,oldCalls+1);
  assert.deepEqual(JSON.parse(stored),expected);
  assert.deepEqual(m.DEFAULT_TIMER_PREFS,defaults);
}
for(const raw of ['bad json','null','{}',null]) {stored=raw;
  m.writeTimerPrefs({break:8});
  assert.deepEqual(JSON.parse(stored),{...defaults,break:8});
}
stored=JSON.stringify({mode:'flow',focus:53,break:2.6});
assert.deepEqual(m.readTimerPrefs(),{mode:'flow',focus:55,break:3});
for(const method of ['getItem','setItem']) {
  globalThis.localStorage={getItem(){if(method==='getItem')throw Error('blocked');
      return null;
    },setItem(){if(method==='setItem')throw Error('quota');
    }};
  assert.deepEqual(m.readTimerPrefs(),defaults);
  assert.doesNotThrow(()=>m.writeTimerPrefs({focus:60}));
}
Object.defineProperty(globalThis,'localStorage',{configurable:true,get(){throw Error('access blocked');
  }});
assert.deepEqual(m.readTimerPrefs(),defaults);
assert.doesNotThrow(()=>m.writeTimerPrefs({focus:60}));
assert.deepEqual(m.normalizeTimerPrefs(null),defaults);
