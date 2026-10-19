import assert from 'node:assert/strict';
const m=await import('./client/src/lib/focusTags.ts');
assert.equal(typeof m.normalizeTagFilter,'function');
const ids=['work','study','creative','reading','planning','other'];
assert.deepEqual(m.FOCUS_TAGS.map(t=>t.id),ids);
assert.deepEqual(m.FOCUS_TAGS.map(t=>t.label),['Work','Study','Creative','Reading','Planning','Other']);
assert.deepEqual(m.TAG_FILTER_OPTIONS.map(t=>t.value),['all','untagged',...ids]);
for(const value of ['all','untagged',...ids]) {assert.equal(m.isTagFilter(value),true);
  assert.equal(m.normalizeTagFilter(value),value);
}
for(const value of [null,undefined,true,1,{},[],'',' work','work ','WORK','__proto__','constructor']) {assert.equal(m.isTagFilter(value),false);
  assert.equal(m.normalizeTagFilter(value),'all');
}
for(const tag of ids){assert.equal(m.isFocusTagId(tag),true);
  assert.equal(m.focusTagLabel(tag),m.FOCUS_TAGS.find(t=>t.id===tag).label);
}
assert.equal(m.matchesTagFilter(null,'all'),true);
assert.equal(m.matchesTagFilter(null,'untagged'),true);
assert.equal(m.matchesTagFilter('work','untagged'),false);
assert.equal(m.matchesTagFilter('work','work'),true);
assert.equal(m.matchesTagFilter('study','work'),false);
for(const value of [1,2,Number.MAX_SAFE_INTEGER])assert.equal(m.isRevision(value),true);
for(const value of [0,-1,0.5,NaN,Infinity,'1',true,null])assert.equal(m.isRevision(value),false);
assert.deepEqual(m.readCalendarTag({}),{kind:'absent'});
assert.deepEqual(m.readCalendarTag({private_tag:'work',private_tag_version:null}),{kind:'malformed'});
assert.deepEqual(m.readCalendarTag({private_tag:'bad',private_tag_version:1}),{kind:'malformed'});
assert.deepEqual(m.readCalendarTag(Object.freeze({private_tag:null,private_tag_version:null})),{kind:'ok',tag:null,version:null});
assert.deepEqual(m.readCalendarTag({private_tag:'work',private_tag_version:2}),{kind:'ok',tag:'work',version:2});
const row=Object.freeze({session_id:'s',user_id:'u',tag:'work',version:1});
assert.deepEqual(m.readConfirmedFocusTag(row,'s','u'),{sessionId:'s',userId:'u',tag:'work',version:1});
assert.equal(m.readConfirmedFocusTag(row,'s','other'),null);
assert.equal(m.readConfirmedFocusTag({...row,version:false},'s','u'),null);
