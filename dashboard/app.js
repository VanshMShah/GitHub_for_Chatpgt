'use strict';
const $=id=>document.getElementById(id);
let lastError=null;
let token=sessionStorage.getItem('atsApiToken')||'', active='overview', refreshing=false, busy=false, paper={}, state={}, ledger=[], preferenceLoaded=false;
const percent=v=>Number.isFinite(Number(v))&&v!==null?(100*Number(v)).toFixed(2)+'%':'—';
const money=v=>v===null||v===undefined?'—':Number(v).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
const date=v=>v?new Date(v).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',hour12:true})+' IST':'—';
const text=(id,value)=>{$(id).textContent=value??'—';};
async function request(path,method='GET',body){
 const headers={};if(token)headers.Authorization='Bearer '+token;
 if(body!==undefined)headers['Content-Type']='application/json';
 const res=await fetch(path,{method,headers,body:body===undefined?undefined:JSON.stringify(body)});
 const data=await res.json();if(!res.ok)throw new Error(data.detail||'Request failed');return data;
}
function notice(message,error=false){if(error)lastError=message;text('notice',message);$('notice').classList.toggle('error',error);}
function table(id,rows,columns,empty='No records yet.'){
 const root=$(id);root.replaceChildren();if(!rows.length){const p=document.createElement('p');p.textContent=empty;root.append(p);return;}
 const wrap=document.createElement('div');wrap.className='tablewrap';const t=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');
 for(const c of columns){const th=document.createElement('th');th.textContent=c[0];tr.append(th);}head.append(tr);t.append(head);
 const body=document.createElement('tbody');for(const row of rows){const r=document.createElement('tr');for(const c of columns){const td=document.createElement('td');td.textContent=c[2]?c[2](row[c[1]],row):(row[c[1]]??'—');r.append(td);}body.append(r);}t.append(body);wrap.append(t);root.append(wrap);
}
function info(id,obj){table(id,Object.entries(obj).map(([key,value])=>({key,value:typeof value==='object'?JSON.stringify(value):String(value??'—')})),[['Item','key'],['Status','value']]);}
function showView(name){active=name;document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==name);document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===name));}
function drawChart(history,latest){
 const svg=$('chart');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
 const el=(tag,attrs,content)=>{const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(content)e.textContent=content;svg.append(e);return e;};
 const prices=history.prices||[];if(!prices.length){el('text',{x:50,y:170,fill:'#9bb2c8'},'Download history to see prices.');return;}
 const f=latest?.forecast,reference=latest?.last_close;const bounds=f?[f.q05,f.q50,f.q95].map(r=>reference*Math.exp(r)):[];
 const all=prices.map(p=>p.close).concat(bounds);let lo=Math.min(...all),hi=Math.max(...all);const pad=Math.max((hi-lo)*.12,1);lo-=pad;hi+=pad;
 const start=new Date(prices[0].event_time).getTime(),end=Math.max(new Date(prices.at(-1).event_time).getTime(),new Date(latest?.target_time||prices.at(-1).event_time).getTime());
 const x=t=>70+(new Date(t).getTime()-start)/Math.max(1,end-start)*890,y=p=>280-(p-lo)/Math.max(1,hi-lo)*250;
 for(let i=0;i<4;i++){const value=lo+(hi-lo)*i/3;el('line',{x1:70,x2:960,y1:y(value),y2:y(value),stroke:'#263b4d'});el('text',{x:2,y:y(value)+4,fill:'#9bb2c8','font-size':12},Math.round(value).toLocaleString('en-IN'));}
 el('polyline',{points:prices.map(p=>`${x(p.event_time)},${y(p.close)}`).join(' '),fill:'none',stroke:'#e3edf7','stroke-width':2});
 if(f&&latest.reference_time){const x0=x(latest.reference_time),x1=x(latest.target_time);el('polygon',{points:`${x0},${y(reference)} ${x1},${y(bounds[2])} ${x1},${y(bounds[0])}`,fill:'#62e3b0',opacity:.15});el('line',{x1:x0,y1:y(reference),x2:x1,y2:y(bounds[1]),stroke:'#62e3b0','stroke-width':3,'stroke-dasharray':'5 4'});el('text',{x:Math.max(70,x1-180),y:320,fill:'#62e3b0','font-size':12},'Forecast target: '+new Date(latest.target_time).toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata'}));}
 el('text',{x:70,y:320,fill:'#9bb2c8','font-size':12},new Date(prices[0].event_time).toLocaleDateString('en-IN'));
}
function renderLedger(){const filter=$('historyFilter').value;table('ledger',ledger.filter(r=>filter==='ALL'||r.state===filter),[['Instrument','symbol'],['Horizon','horizon'],['Created','created_at',date],['Target','target_time',date],['State','state'],['P(up)','p_up',percent],['Estimate','mean',percent],['Actual','target_return',percent],['Absolute error','abs_error',percent]],'No predictions yet. Run Sync or Daily forecast.');}
function render(s,h,performance,models,account,settings,diagnostics){
 state=s;paper=account;const l=s.latest_daily||s.latest,f=l?.forecast;
 text('subtitle',`Daily NIFTY research · ${s.market_hours?'Market open':'Off market'} · ${s.worker?.running?'Worker running':'Worker stopped'}`);
 text('spot',money(s.spot));text('source',`${s.spot_source||'No quote'} · observed ${date(s.spot_time)}`);
 text('direction',f?(f.p_up>=.5?'Upward':'Downward'):'No forecast');text('prob',f?`P(up) ${percent(f.p_up)} · P(down) ${percent(1-f.p_up)}`:'Run Sync to download, train and forecast.');
 text('decision',l?`${l.decision?.bias||'WAIT'} · ${l.decision?.reason||''}`:'');text('agreement',l?percent(l.conviction):'—');text('target',date(l?.target_time));text('reference',`Reference session: ${l?.reference_time?new Date(l.reference_time).toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata'}):'—'}`);
 text('estimate',f?percent(f.mean):'—');text('range',f?`${percent(f.q05)} to ${percent(f.q95)}`:'—');text('vol',f?percent(f.volatility):'—');text('var',f?percent(Math.max(0,-f.q05)):'—');
 $('weights').replaceChildren();for(const [name,weight] of Object.entries(l?.weights||{})){const row=document.createElement('div');row.className='weight';const label=document.createElement('span');label.textContent=name+' · '+percent(weight);const bar=document.createElement('progress');bar.value=weight;bar.max=1;row.append(label,bar);$('weights').append(row);}
 text('weightsLabel',s.policy_weight_source==='learned'?'Learned from settled daily outcomes':'Initial prior · awaiting settled outcomes');text('riskNote',`${s.risk_profile?.level||'medium'} profile · minimum expert agreement ${percent(s.risk_profile?.min_conviction)}`);
 document.querySelectorAll('[data-risk]').forEach(b=>b.classList.toggle('selected',b.dataset.risk===s.risk_profile?.level));
 table('scores',performance.groups||[],[['Instrument','symbol'],['Horizon','horizon'],['Settled','n'],['Direction accuracy','directional_accuracy',percent],['MAE','mae',percent],['Brier','brier',money],['Log loss','log_loss',money],['90% coverage','coverage90',percent],['Empirical tail loss','empirical_tail_loss95',percent]],'No settled outcomes yet. Pending forecasts will be scored after their target data arrives.');
 if(performance.benchmark){table('benchmark',Object.entries(performance.benchmark.models).map(([name,v])=>({name,...v})),[['Model','name'],['Direction accuracy','directional_accuracy',percent],['Brier','brier',money],['Log loss','log_loss',money],['MAE','mean_absolute_error',percent]]);}else text('benchmark','No benchmark run recorded for this installation.');
 info('modelState',{'Daily training rows':models.daily?.training_rows,'Latest labeled training row':date(models.daily?.training_end),'Retrain pending':models.daily?.pending,'Daily neural champion':models.daily_neural?.champion_exists,'Daily challenger trigger':models.daily_neural?.trigger,'Intraday instrument':models.instrument,'One-minute rows':models.intraday_rows,'Intraday champion file':models.intraday_champion,'Intraday champion metadata':models.intraday,'Latest intraday evaluation':models.intraday_last_run});
 text('paperState',`${account.running?'Runner running':'Runner stopped'} · ${account.halted?'KILL SWITCH ACTIVE':'Kill switch clear'} · ${account.heartbeat?.state||'Not started'} · ${account.heartbeat?.reason||''}`);
 $('paperSummary').replaceChildren();for(const [name,value]of [['Equity',account.equity],['Net realized P/L',account.net_realized_pnl],['Unrealized P/L',account.unrealized_pnl],['Exposure',account.notional]]){const d=document.createElement('div');d.textContent=name;const strong=document.createElement('strong');strong.textContent='₹'+money(value);d.append(strong);$('paperSummary').append(d);}
 table('positions',account.positions||[],[['Instrument','symbol'],['Quantity','quantity'],['Average entry','average_price',money],['Last mark','mark',money],['Net realized','net_realized_pnl',money],['Unrealized','unrealized_pnl',money],['Marked at','updated_at',date]],'No paper positions.');
 table('orders',account.orders||[],[['Order','order_id'],['Instrument','symbol'],['Side','side'],['Requested','quantity'],['Filled','filled_quantity'],['Status','status'],['Reason','reason'],['Price','fill_price',money],['Fee','fee',money]]);
 table('fills',account.fills||[],[['Order','order_id'],['Side','side'],['Quantity','quantity'],['Price','price',money],['Fee','fee',money],['Time','timestamp_ns',v=>date(Number(v)/1e6)]]);
 text('costNote',`${account.cost_note||''} Combined rate ${account.fee_bps||0} bps + ₹${account.fixed_fee||0} per fill.`);
 const feeds=Object.entries(s.feeds||{}).map(([name,v])=>({name,status:v?.ok?'OK':'FAILED / UNAVAILABLE',source:v?.source,time:v?.updated_at,error:v?.error}));
 table('healthState',feeds,[['Feed','name'],['Status','status'],['Source','source'],['Updated','time',date],['Error','error']]);
 const worker=document.createElement('p');worker.textContent=`Worker last success: ${date(s.worker?.last_success)} · last failure: ${JSON.stringify(s.worker?.last_failure||'none')}`;$('healthState').append(worker);
 table('diagnosticState',diagnostics,[['Time','timestamp',date],['Code','code'],['Action','action'],['Details','detail_json']]);
 info('configState',{'Reference instrument':settings.reference_instrument,'Execution instrument':settings.instrument||'Not configured','Upstox token present':settings.token_present,'Paper enabled':settings.paper_enabled,'API authentication':settings.api_auth_enabled,'Lot size':settings.lot_size,'Shorting permitted in simulator':settings.allow_short,'Database':settings.db_path,'Session policy':settings.paper_policy});
 if(!preferenceLoaded){preferenceLoaded=true;if(settings.preferences?.default_view)showView(settings.preferences.default_view);}
 const running=['RUNNING','QUEUED'].includes(s.job?.status);$('start').disabled=!!s.worker?.running||running||account.running;$('stop').disabled=!s.worker?.running;document.querySelectorAll('[data-job]').forEach(b=>b.disabled=running||s.worker?.running||account.running||busy);
 text('job',s.job?.id?`${s.job.kind}: ${s.job.status}${s.job.error?' — '+s.job.error:''}${s.job.status==='SUCCEEDED'?' · Completed; saved result is available below.':''}`:'');$('job').classList.toggle('error',s.job?.status==='FAILED');
 drawChart(h,l);
}
async function refresh(){if(refreshing)return;refreshing=true;try{
 const [s,h,l,p,m,a,c,d]=await Promise.all(['/api/status','/api/history','/api/forecasts','/api/performance','/api/models','/api/paper','/api/settings','/api/diagnostics'].map(path=>request(path)));
 ledger=l;render(s,h,p,m,a,c,d);renderLedger();
 if(!busy&&!lastError)notice(`${s.feed_mode==='DEMO'?'OFFLINE DEMO — SYNTHETIC DATA':s.feed_mode==='LIVE_BROKER'?'Upstox market data':'Fallback market data'} · ${s.feed_stale?'Quote stale':'check observation timestamp'} · Daily forecasts ${s.forecasts}, settled ${s.settled}. Paper fills are local simulations.`);
 }catch(e){notice(e.message,true);}finally{refreshing=false;}}
async function action(path,body){if(busy)return;busy=true;lastError=null;notice('Working…');try{const result=await request(path,'POST',body);notice(result.status?`Request: ${result.status}`:JSON.stringify(result));}catch(e){notice(e.message,true);}finally{busy=false;await refresh();}}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>showView(b.dataset.view));
document.querySelectorAll('[data-job]').forEach(b=>b.onclick=()=>action('/api/jobs',{kind:b.dataset.job}));
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>action(b.dataset.action));
document.querySelectorAll('[data-risk]').forEach(b=>b.onclick=()=>action('/api/risk',{level:b.dataset.risk}));
document.querySelectorAll('[data-export]').forEach(b=>b.onclick=async()=>{try{const r=await fetch('/api/export/'+b.dataset.export,{headers:token?{Authorization:'Bearer '+token}:{}});if(!r.ok)throw new Error('Export failed');const blob=await r.blob();const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='ats_'+b.dataset.export+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice(e.message,true);}});
$('historyFilter').onchange=renderLedger;$('halt').onclick=()=>action('/api/paper/kill',{halt:!paper.halted});
$('saveToken').onclick=()=>{token=$('apiToken').value;sessionStorage.setItem('atsApiToken',token);$('apiToken').value='';refresh();};
$('savePrefs').onclick=()=>action('/api/settings',{risk_level:state.risk_profile?.level||'medium',default_view:$('defaultView').value});
setInterval(refresh,5000);refresh();
