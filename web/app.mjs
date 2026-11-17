import { exercises, RepCoach, skeleton } from './coach.mjs';

const paths = {
  pulse:'<path d="M2 12h5l3-7 4 14 3-7h5"/>',
  lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 1 1 8 0v3m-4 5v2"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
  arrow:'<path d="M4 12h15m-6-6 6 6-6 6"/>',
  back:'<path d="M20 12H5m6 6-6-6 6-6"/>',
  video:'<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/>',
  frame:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="12" r="3"/>',
  stop:'<rect x="6" y="6" width="12" height="12" rx="2"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',
  squats:'<circle cx="13" cy="5" r="2"/><path d="m12 8-4 6h7l-2 7h4m-6-10 4 1 4-2m-9 4 3 3-2 4H8"/>',
  pushups:'<circle cx="5" cy="9" r="2"/><path d="m7 12 6 1 6 3 2-1M8 12l2 7H6m8-5 6 5h2"/>',
  pullups:'<path d="M3 3h18m-6 7 3-2V3M9 10 6 8V3m3 4-1 5 4 2 4-2-1-5m-3 7v3l-2 4m2-4 2 4"/><circle cx="12" cy="7" r="2"/>',
  situp:'<circle cx="5" cy="8" r="2"/><path d="m6 11 4 7 5-6 4 7h3M8 13l5 2m-10 6h19"/>',
  jumping_jacks:'<circle cx="12" cy="6" r="2"/><path d="m5 2 3 7 4 2 4-2 3-7m-7 9v5m0 0-5 6m5-6 5 6"/>'
};

function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
}
const $ = id => document.getElementById(id);
const app = $('app');
let selected = 0;
let page = 'pick';
let session = null;
let detector = null;
let detectorLoading = null;
const current = () => exercises[selected];
const timecode = seconds => {
  const whole = Math.max(0, Math.floor(seconds));
  return [Math.floor(whole / 60), whole % 60].map(number => String(number).padStart(2, '0')).join(':');
};

for (const node of document.querySelectorAll('[data-icon]')) node.innerHTML = icon(node.dataset.icon);
$('home').onclick = goHome;
window.addEventListener('pagehide', stopCapture);
window.addEventListener('pageshow', event => {
  if (event.persisted && page === 'live') {
    if (session.mode) finish();
    else renderLive();
  }
});

function switchPage(next, title) {
  page = next;
  const active = ['pick','live','summary'].indexOf(next);
  $('steps').innerHTML = '<div class="steps">' + ['Exercise','Live workout','Summary'].map((label, index) =>
    `<span class="step ${index === active ? 'current' : index < active ? 'past' : ''}" ${index === active ? 'aria-current="step"' : ''}><i>${index < active ? icon('check') : index + 1}</i>${label}</span>`
  ).join('<span class="step-line" aria-hidden="true"></span>') + '</div>';
  document.title = `${title} · Repform`;
}

function focusTitle() {
  app.querySelector('h1')?.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

function goHome() {
  if (page === 'live' && session.mode) return finish();
  stopCapture();
  session = null;
  renderPicker();
  focusTitle();
}

function renderPicker() {
  switchPage('pick', 'Choose your exercise');
  app.innerHTML = `
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-content">
        <p class="eyebrow"><i class="dot"></i> LIVE FORM COACH</p>
        <h1 id="hero-title" tabindex="-1">Train with<br><span>better form.</span></h1>
        <p>Get real-time coaching cues and automatic rep counts using your camera.</p>
      </div>
    </section>
    <section class="finder" aria-labelledby="picker-title">
      <div class="section-head">
        <div><h2 id="picker-title">Choose your exercise</h2><p>Pick one movement to start your workout.</p></div>
      </div>
      <div class="exercise-grid" role="radiogroup" aria-labelledby="picker-title">
        ${exercises.map((exercise,index) => `
          <button class="exercise" type="button" role="radio" data-exercise="${index}">
            <span class="tile-icon">${icon(exercise.id)}</span><span class="select-dot"></span>
            <h3>${exercise.name}</h3><small>${exercise.group}</small>
          </button>`).join('')}
      </div>
      <div class="setup">
        <span class="tile-icon">${icon('frame')}</span>
        <div class="setup-text" aria-live="polite"><strong id="setup-view"></strong><p id="setup-tip"></p></div>
        <button class="primary" id="continue" type="button">Continue to camera ${icon('arrow')}</button>
      </div>
    </section>`;

  const buttons = [...app.querySelectorAll('.exercise')];
  function choose(index, focus = false) {
    selected = index;
    for (const [key, button] of buttons.entries()) {
      button.setAttribute('aria-checked', String(key === index));
      button.tabIndex = key === index ? 0 : -1;
      button.querySelector('.select-dot').innerHTML = key === index ? icon('check') : '';
    }
    $('setup-view').textContent = `Recommended camera angle · ${current().view} view`;
    $('setup-tip').textContent = current().tip;
    if (focus) buttons[index].focus();
  }
  buttons.forEach((button,index) => {
    button.onclick = () => choose(index);
    button.onkeydown = event => {
      const direction = {ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[event.key];
      if (!direction) return;
      event.preventDefault();
      choose((index + direction + buttons.length) % buttons.length, true);
    };
  });
  choose(selected);
  $('continue').onclick = begin;
}

function begin() {
  session = { mode:null, attempt:0, events:[], reps:0, reviewed:0, started:0, duration:null, stream:null, loop:0, timer:0 };
  renderLive();
  focusTitle();
}

function historyMarkup() {
  return `<section class="history" aria-label="Feedback history">
    <div class="history-head"><h2>Feedback history</h2><span class="history-help">ELAPSED TIME</span></div>
    <ol class="history-list" id="events" aria-live="polite" aria-relevant="additions"></ol>
  </section>`;
}

function renderLive() {
  switchPage('live','Live workout');
  app.innerHTML = `
    <div class="page-title">
      <div><div class="eyebrow"><i class="dot"></i> Live workout</div>
      <h1 tabindex="-1">${current().name}</h1><p>Recommended camera angle · ${current().view} view</p></div>
      <button class="secondary" id="exit" type="button">${icon('back')} Exercises</button>
    </div>
    <div class="session-grid">
      <div class="main-stack">
        <div class="camera">
          <video id="video" autoplay playsinline muted aria-label="Mirrored camera preview"></video>
          <canvas id="landmarks" aria-hidden="true"></canvas>
          <span class="view-tag"><i class="dot"></i><span id="camera-tag">CAMERA OFF</span></span>
          <span class="tracking-tag" id="tracking-tag">READY</span>
          <div class="demo-mark" id="demo-mark" hidden aria-hidden="true">${icon(current().id)}</div>
          <div class="camera-empty" id="camera-empty">
            ${icon('video')}
            <h2 id="camera-title">Set up your camera</h2>
            <p id="camera-copy" role="status">${current().tip}</p>
            <div class="camera-buttons">
              <button class="primary" id="enable" type="button">Enable camera</button>
              <button class="quiet" id="demo" type="button">Try demo</button>
            </div>
          </div>
        </div>
        <section class="cue" id="coach" role="status" aria-atomic="true">
          <div class="eyebrow" id="cue-label"><i class="dot"></i> Coach feedback</div>
          <h2 id="cue-title">Get ready to move</h2><p id="cue-copy">Your coach will show one cue at a time here.</p>
        </section>
      </div>
      <div class="side-stack">
        <div class="metrics"><div class="metric"><small>Reps completed</small><strong id="reps">0</strong></div><div class="metric"><small>Elapsed time</small><strong id="clock">00:00</strong></div></div>
        ${historyMarkup()}
        <p class="legend">Form corrections are highlighted in orange.</p>
        <button class="primary" id="finish" type="button" hidden>${icon('stop')} Finish workout</button>
      </div>
    </div>`;
  $('exit').onclick = goHome;
  $('enable').onclick = enableCamera;
  $('demo').onclick = startDemo;
  $('finish').onclick = finish;
}

function setCue(title, description, correction = false) {
  const block = $('coach');
  if (!block || ($('cue-title').textContent === title && $('cue-copy').textContent === description)) return;
  block.classList.toggle('correction', correction);
  $('cue-label').innerHTML = `<i class="dot"></i> ${correction ? 'Form correction' : 'Coach feedback'}`;
  $('cue-title').textContent = title;
  $('cue-copy').textContent = description;
}

function addHistoryRow(event, prepend) {
  const item = document.createElement('li');
  item.className = `event ${event.kind}`;
  const time = document.createElement('time');
  time.textContent = timecode(event.t);
  time.dateTime = `PT${Math.floor(event.t)}S`;
  const message = document.createElement('div');
  const label = document.createElement('small');
  label.textContent = `Rep ${event.rep} · ${event.label}`;
  const title = document.createElement('b');
  title.textContent = event.text;
  message.append(label,title);
  item.append(time,message);
  const list = $('events');
  if (prepend) list.prepend(item);
  else list.append(item);
}

function logResult(result, elapsed) {
  session.reps = result.rep;
  if (result.checks) session.reviewed++;
  $('reps').textContent = session.reps;
  const events = result.issues.length ? result.issues.map(issue => ({
    t:elapsed, rep:result.rep, label:issue.label,
    text:issue.text || issue.issue,
    kind:'correction'
  })) : [{
    t:elapsed, rep:result.rep, label:result.checks ? 'Form check' : 'Rep count',
    text:result.checks ? 'No corrections detected' : 'Rep counted · form unavailable',
    kind:'note'
  }];
  // Insert in reverse so the strongest correction sits at the top of the group.
  for (const event of [...events].reverse()) {
    session.events.push(event);
    addHistoryRow(event,true);
  }
  setCue(events[0].text, `Rep ${result.rep} · ${events[0].label}`, events[0].kind === 'correction');
  session.holdUntil = elapsed + 3;
}

async function getDetector() {
  if (detector) return detector;
  if (!detectorLoading) {
    const library = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
    detectorLoading = (async () => {
      const { FilesetResolver, PoseLandmarker } = await import(library + '/vision_bundle.mjs');
      const runtime = await FilesetResolver.forVisionTasks(library + '/wasm');
      return PoseLandmarker.createFromOptions(runtime, {
        baseOptions:{
          modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        },
        runningMode:'VIDEO', numPoses:1
      });
    })().then(instance => (detector = instance)).catch(error => {
      detectorLoading = null;
      throw error;
    });
  }
  return detectorLoading;
}

async function enableCamera() {
  const active = session;
  const attempt = ++active.attempt;
  const stillCurrent = () => page === 'live' && session === active && attempt === active.attempt;
  const video = $('video');
  $('enable').disabled = true;
  $('camera-title').textContent = 'Starting camera';
  $('camera-copy').textContent = 'Allow camera access when your browser asks.';
  $('tracking-tag').textContent = 'CONNECTING';
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Open the site on localhost or HTTPS in a browser with camera support.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:960},height:{ideal:540},facingMode:'user'},
      audio:false
    });
    if (!stillCurrent()) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    active.stream = stream;
    for (const track of stream.getVideoTracks()) track.addEventListener('ended', () => {
      if (stillCurrent()) cameraProblem('The camera disconnected. Reconnect it and try again.');
    });
    video.srcObject = stream;
    await video.play();
    if (!stillCurrent()) return;
    $('camera-copy').textContent = 'Loading the pose model. The first load can take a moment.';
    let model;
    try {
      model = await getDetector();
    } catch (error) {
      console.error(error);
      throw new Error('The pose model could not load. Check your internet connection and try again.');
    }
    if (!stillCurrent()) return;
    video.classList.add('visible');
    activate('camera');
    active.coach = new RepCoach(current().id);
    setCue('Get into position', 'The workout starts in three seconds.');
    let lastFrame = -1;
    let lastInference = 0;
    const loop = stamp => {
      if (!stillCurrent()) return;
      active.loop = requestAnimationFrame(loop);
      if (stamp - lastInference < 48 || video.currentTime === lastFrame || video.readyState < 2) return;
      lastFrame = video.currentTime;
      lastInference = stamp;
      const elapsed = (stamp - active.started) / 1000;
      try {
        const { landmarks } = model.detectForVideo(video,stamp);
        drawPose(landmarks[0], video.videoWidth, video.videoHeight);
        if (elapsed < 0) {
          setCue(`Starting in ${Math.ceil(-elapsed)}`,current().tip);
          return;
        }
        const state = active.coach.update(elapsed,landmarks[0],video.videoWidth,video.videoHeight);
        $('tracking-tag').textContent = state.seen ? 'POSE DETECTED' : 'STEP INTO FRAME';
        if (state.result) logResult(state.result,elapsed);
        if (elapsed < (active.holdUntil ?? 0)) return;
        if (!state.seen) setCue('Step fully into frame',current().tip);
        else if (state.phase === 'waiting') setCue(current().ready,'Complete the movement and return to the starting position.');
        else if (state.phase === 'rest') setCue('Ready for your next rep','Move at a steady pace.');
        else setCue('Return to start','Finish the movement to count this rep.');
      } catch (error) {
        console.error(error);
        cameraProblem('Tracking stopped unexpectedly. Try reconnecting your camera.');
      }
    };
    active.loop = requestAnimationFrame(loop);
  } catch (error) {
    if (!stillCurrent()) return;
    const messages = {
      NotAllowedError:'Camera access was denied. Allow it in your browser settings and try again.',
      NotFoundError:'No camera was found. Connect a camera, or try the demo.',
      NotReadableError:'The camera is busy. Close other apps using it and try again.'
    };
    cameraProblem(messages[error.name] || error.message || 'Could not start the camera.');
  }
}

function cameraProblem(message) {
  const hadWorkout = Boolean(session.mode);
  stopCapture();
  $('camera-empty').hidden = false;
  $('camera-title').textContent = 'Camera unavailable';
  $('camera-copy').textContent = message;
  $('camera-tag').textContent = 'CAMERA OFF';
  $('tracking-tag').textContent = 'OFFLINE';
  $('enable').disabled = false;
  if (hadWorkout) {
    session.duration = Math.max(0,(performance.now() - session.started) / 1000);
    $('enable').hidden = true;
    $('demo').hidden = true;
    setCue('Tracking stopped', 'Finish the workout to review your reps and feedback.');
  }
}

function activate(mode) {
  session.mode = mode;
  session.started = performance.now() + (mode === 'camera' ? 3000 : 0);
  $('camera-empty').hidden = true;
  $('camera-tag').textContent = mode === 'camera' ? 'LIVE CAMERA' : 'DEMO · CAMERA OFF';
  $('tracking-tag').textContent = mode === 'camera' ? 'WARMING UP' : 'SAMPLE WORKOUT';
  $('exit').hidden = true;
  $('finish').hidden = false;
  $('finish').focus({preventScroll:true});
  session.timer = setInterval(() => {
    $('clock').textContent = timecode((performance.now() - session.started) / 1000);
  }, 250);
}

function drawPose(points, width, height, color = '#89bcff') {
  const canvas = $('landmarks');
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,width,height);
  if (!points) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2,width/260);
  ctx.lineCap = 'round';
  for (const [a,b] of skeleton) {
    const p = points[a], q = points[b];
    if ((p?.visibility ?? 0) < .5 || (q?.visibility ?? 0) < .5) continue;
    ctx.beginPath();
    ctx.moveTo(p.x*width,p.y*height);
    ctx.lineTo(q.x*width,q.y*height);
    ctx.stroke();
  }
  ctx.fillStyle = '#e1edff';
  for (const p of points) {
    if ((p?.visibility ?? 0) < .5) continue;
    ctx.beginPath();
    ctx.arc(p.x*width,p.y*height,Math.max(2.4,width/220),0,2*Math.PI);
    ctx.fill();
  }
}

const demoCorrection = {
  squats:['Squat depth','Lower your hips a little farther'],
  pushups:['Depth','Lower your chest a little farther'],
  pullups:['Chin over the bar','Bring your chin higher'],
  situp:['Sit up to vertical','Sit up a little taller'],
  jumping_jacks:['Arms overhead','Reach your arms overhead']
};

function startDemo() {
  stopCapture();
  activate('demo');
  setCue('Watch a sample rep','Demo feedback uses sample data.');
  const active = session;
  const [label,text] = demoCorrection[current().id];
  active.timerDemo = setTimeout(() => logResult({rep:1,checks:1,issues:[{label,text}]},2),2000);
  active.timerDemoEnd = setTimeout(() => logResult({rep:2,checks:1,issues:[]},6),6000);
  $('demo-mark').hidden = false;
}

function stopCapture() {
  if (!session) return;
  ++session.attempt;
  cancelAnimationFrame(session.loop);
  clearInterval(session.timer);
  clearTimeout(session.timerDemo);
  clearTimeout(session.timerDemoEnd);
  session.stream?.getTracks().forEach(track => track.stop());
  session.stream = null;
  const video = $('video');
  if (video) {
    video.srcObject = null;
    video.classList.remove('visible');
  }
  const canvas = $('landmarks');
  if (canvas) canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
}

function finish() {
  if (!session?.mode) return;
  if (session.duration === null) session.duration = Math.max(0,(performance.now() - session.started) / 1000);
  stopCapture();
  renderSummary();
  focusTitle();
}

function renderSummary() {
  switchPage('summary','Workout summary');
  const corrections = session.events.filter(event => event.kind === 'correction');
  const counts = new Map();
  for (const correction of corrections) counts.set(correction.text,(counts.get(correction.text) || 0) + 1);
  const top = [...counts.entries()].sort((a,b) => b[1]-a[1])[0];
  app.innerHTML = `
    <section class="done">
      <div><div class="eyebrow"><i class="dot"></i> ${session.mode === 'demo' ? 'Demo summary' : 'Workout summary'}</div>
      <h1 tabindex="-1">Workout complete.</h1><p>${current().name} · ${session.mode === 'demo' ? 'Sample workout' : 'Camera session'}</p></div>
      <span class="done-icon">${icon('check')}</span>
    </section>
    <div class="stats">
      <div class="stat"><small>Reps completed</small><strong>${session.reps}</strong></div>
      <div class="stat"><small>Workout time</small><strong>${timecode(session.duration)}</strong></div>
      <div class="stat"><small>Form corrections</small><strong>${corrections.length}</strong></div>
    </div>
    <div class="summary-grid">
      <section class="focus">
        <div class="eyebrow">Next workout</div>
        <h2 id="focus-title"></h2><p id="focus-copy"></p>
      </section>
      ${historyMarkup()}
    </div>
    <div class="summary-actions">
      <button class="primary" id="again" type="button">Choose another exercise ${icon('arrow')}</button>
      <button class="secondary" id="download" type="button">${icon('download')} Download feedback</button>
    </div>`;
  $('focus-title').textContent = top ? top[0] : session.reviewed ? 'No corrections detected' : 'Set up your camera';
  $('focus-copy').textContent = top
    ? `This was your most frequent cue. It appeared ${top[1]} ${top[1] === 1 ? 'time' : 'times'}.`
    : session.reviewed
      ? `Form was checked on ${session.reviewed} of ${session.reps} completed reps.`
      : `No form checks were completed. ${current().tip}`;
  for (const event of [...session.events].reverse()) addHistoryRow(event,false);
  $('events').setAttribute('aria-live','off');
  $('again').onclick = goHome;
  $('download').disabled = !session.events.length;
  $('download').onclick = downloadFeedback;
}

function downloadFeedback() {
  const rows = [['exercise','mode','elapsed_seconds','rep','kind','check','feedback'], ...session.events.map(event =>
    [current().name,session.mode,event.t.toFixed(2),event.rep,event.kind,event.label,event.text]
  )];
  const csv = rows.map(row => row.map(value => '"' + String(value).replaceAll('"','""') + '"').join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = `repform-${current().id}-feedback.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

renderPicker();
