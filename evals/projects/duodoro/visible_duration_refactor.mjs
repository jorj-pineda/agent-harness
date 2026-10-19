import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const m=require(process.cwd()+'/server/payloadParsers.js');
assert.equal(m.boundedDuration(60.5,1500,60,7200),60.5);
