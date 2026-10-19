import assert from 'node:assert/strict';
const {unfinishedIntentions:f}=await import('./client/src/lib/sessionWrapUp.ts');
function freeze(v){if(v&&typeof v==='object'){Object.values(v).forEach(freeze);
    Object.freeze(v);
  }return v;
}
for(const phase of ['focus','waiting','ready','break','ended'])for(const currentDone of [false,true])for(const previousDone of [false,true]) {
  const intentions=freeze({current:{u:{text:'  same  ',completed:currentDone},other:{text:'unrelated',completed:false}},next:{u:' next ',other:'other'}});
  const recap=freeze({intentions:{u:{text:'same',completed:previousDone},other:{text:'unrelated',completed:false}}});
  const before=JSON.stringify({intentions,recap});
  const expected=[];
  if(phase==='focus'&&!currentDone)expected.push('same');
  if(!previousDone&&!expected.includes('same'))expected.push('same');
  expected.push('next');
  const result=f('u',phase,intentions,recap);
  assert.deepEqual(result,expected);
  result.push('result-only');
  assert.deepEqual(f('u',phase,intentions,recap),expected);
  assert.equal(JSON.stringify({intentions,recap}),before);
}
for(const text of ['', '  ', '\t\n'])assert.deepEqual(f('u','focus',freeze({current:{u:{text,completed:false}},next:{u:text}}),freeze({intentions:{u:{text,completed:false}}})),[]);
assert.deepEqual(f('u','focus',{current:{u:{text:' Work ',completed:false}},next:{u:'Work'}},{intentions:{u:{text:'work',completed:false}}}),['Work','work']);
assert.deepEqual(f('u','focus',{current:{},next:{}},null),[]);
assert.deepEqual(f('missing','focus',{current:{u:{text:'x',completed:false}},next:{u:'y'}},{intentions:{u:{text:'z',completed:false}}}),[]);
